"""Server for serving a policy over websockets.

Adapted from https://github.com/robo-arena/roboarena/

"""


import asyncio
import dataclasses
import logging
import traceback
from typing import Any

from openpi_client.base_policy import BasePolicy
from openpi_client import msgpack_numpy
import websockets.asyncio.server
import websockets.frames


def _normalize_timeout(timeout_seconds: float | None) -> float | None:
    if timeout_seconds is None:
        return None
    if timeout_seconds <= 0:
        return None
    return timeout_seconds


@dataclasses.dataclass
class PolicyServerConfig:
    # Resolution that images get resized to client-side, None means no resizing.
    # It's beneficial to resize images to the desired resolution client-side for faster communication.
    image_resolution: tuple[int, int] | None = (224, 224)
    # Whether or not wrist camera image(s) should be sent.
    needs_wrist_camera: bool = True
    # Number of external cameras to send.
    n_external_cameras: int = 1  # can be in [0, 1, 2]
    # Whether or not stereo camera image(s) should be sent.
    needs_stereo_camera: bool = False
    # Whether or not the unique eval session id should be sent (e.g. for policies that want to keep track of history).
    needs_session_id: bool = False
    # Which action space to use.
    action_space: str = "joint_position"  # can be in ["joint_position", "joint_velocity", "cartesian_position", "cartesian_velocity"]
    # Optional DreamZero cache-mode metadata. Clients use these to avoid request
    # patterns that would reorder causal KV/cache updates.
    wam_architecture: str | None = None
    mot_inference_video_mode: str | None = None
    cache_order_sensitive: bool = False
    supports_rtc: bool = True
    supports_async_prefetch: bool = True
    supports_parallel_sessions: bool = False
    supports_batching: bool = False
    max_batch_size: int = 8
    batch_timeout_ms: float = 2.0


@dataclasses.dataclass
class _InferRequest:
    obs: dict[str, Any]
    future: asyncio.Future


@dataclasses.dataclass
class _ResetRequest:
    obs: dict[str, Any]
    future: asyncio.Future


_PolicyRequest = _InferRequest | _ResetRequest


class WebsocketPolicyServer:
    """
    Serves a policy using the websocket protocol.

    Interface:
      Observation:
        - observation/wrist_image_left: (H, W, 3) if needs_wrist_camera is True
        - observation/wrist_image_right: (H, W, 3) if needs_wrist_camera is True and needs_stereo_camera is True
        - observation/exterior_image_{i}_left: (H, W, 3) if n_external_cameras >= 1
        - observation/exterior_image_{i}_right: (H, W, 3) if needs_stereo_camera is True
        - session_id: (1,) if needs_session_id is True
        - observation/joint_position: (7,)
        - observation/cartesian_position: (6,)
        - observation/gripper_position: (1,)
        - prompt: str, the natural language task instruction for the policy
    
      Action:
        - action: (N, 8,) or (N, 7,): either 7 movement actions (for joint action spaces) or 6 (for cartesian) plus one dimension for gripper position
                           --> all N actions will get executed on the robot before the server is queried again
        - policies may return either a raw action array or a dict containing
          {"actions": action_array}; the websocket server normalizes raw arrays
          to the RoboArena-compatible dict response format.

    """

    def __init__(
        self,
        policy: BasePolicy,
        server_config: PolicyServerConfig,
        host: str = "0.0.0.0",
        port: int = 8000,
        open_timeout: float | None = 0.0,
    ) -> None:
        self._policy = policy
        self._server_config = server_config
        self._host = host
        self._port = port
        self._open_timeout = _normalize_timeout(open_timeout)
        self._policy_lock = asyncio.Lock()
        self._request_queue: asyncio.Queue[_PolicyRequest] | None = None
        self._deferred_request: _PolicyRequest | None = None
        self._batch_worker_task: asyncio.Task | None = None
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        if self._server_config.supports_batching:
            self._request_queue = asyncio.Queue()
            self._batch_worker_task = asyncio.create_task(self._infer_batch_worker())
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            open_timeout=self._open_timeout,
            ping_interval=None,
        ) as server:
            try:
                await server.serve_forever()
            finally:
                if self._batch_worker_task is not None:
                    self._batch_worker_task.cancel()

    async def _infer(self, obs: dict[str, Any]) -> dict:
        if not self._server_config.supports_batching:
            async with self._policy_lock:
                action = self._policy.infer(obs)
            if not isinstance(action, dict):
                action = {"actions": action}
            return action

        if self._request_queue is None:
            raise RuntimeError("Batched inference queue was not initialized.")

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self._request_queue.put(_InferRequest(obs=obs, future=future))
        return await future

    async def _reset(self, obs: dict[str, Any]) -> str:
        if not self._server_config.supports_batching:
            async with self._policy_lock:
                self._policy.reset(obs)
            return "reset successful"

        if self._request_queue is None:
            raise RuntimeError("Batched inference queue was not initialized.")

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self._request_queue.put(_ResetRequest(obs=obs, future=future))
        return await future

    async def _infer_batch_worker(self) -> None:
        assert self._request_queue is not None
        max_batch_size = max(int(self._server_config.max_batch_size), 1)
        batch_timeout = max(float(self._server_config.batch_timeout_ms), 0.0) / 1000.0

        while True:
            if self._deferred_request is not None:
                first_request = self._deferred_request
                self._deferred_request = None
            else:
                first_request = await self._request_queue.get()

            if isinstance(first_request, _ResetRequest):
                try:
                    async with self._policy_lock:
                        self._policy.reset(first_request.obs)
                    if not first_request.future.cancelled():
                        first_request.future.set_result("reset successful")
                except Exception as exc:
                    if not first_request.future.cancelled():
                        first_request.future.set_exception(exc)
                continue

            active_requests = [first_request]
            try:
                requests = [first_request]
                deadline = asyncio.get_running_loop().time() + batch_timeout
                while len(requests) < max_batch_size:
                    if batch_timeout == 0:
                        try:
                            request = self._request_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    else:
                        timeout = deadline - asyncio.get_running_loop().time()
                        if timeout <= 0:
                            break
                        try:
                            request = await asyncio.wait_for(self._request_queue.get(), timeout=timeout)
                        except asyncio.TimeoutError:
                            break
                    if isinstance(request, _ResetRequest):
                        self._deferred_request = request
                        break
                    requests.append(request)

                active_requests = [request for request in requests if not request.future.cancelled()]
                if not active_requests:
                    continue

                async with self._policy_lock:
                    if len(active_requests) > 1 and hasattr(self._policy, "infer_many"):
                        actions = self._policy.infer_many([request.obs for request in active_requests])
                    else:
                        actions = [self._policy.infer(request.obs) for request in active_requests]

                if len(actions) != len(active_requests):
                    raise RuntimeError(
                        "Policy infer_many returned "
                        f"{len(actions)} actions for {len(active_requests)} observations."
                    )

                for request, action in zip(active_requests, actions, strict=True):
                    if not isinstance(action, dict):
                        action = {"actions": action}
                    if not request.future.cancelled():
                        request.future.set_result(action)
            except Exception as exc:
                for request in active_requests:
                    if not request.future.cancelled():
                        request.future.set_exception(exc)
                if self._deferred_request is not None and self._deferred_request.future.cancelled():
                    self._deferred_request = None


    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        # Send server config to client to configure what gets sent to server.
        await websocket.send(packer.pack(dataclasses.asdict(self._server_config)))

        while True:
            try:
                obs = msgpack_numpy.unpackb(await websocket.recv())
                
                endpoint = obs["endpoint"]
                del obs["endpoint"]
                if endpoint == "reset":
                    to_return = await self._reset(obs)
                else:
                    action = await self._infer(obs)
                    to_return = packer.pack(action)
                await websocket.send(to_return)
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


if __name__ == "__main__":
    import numpy as np

    class DummyPolicy(BasePolicy):
        def infer(self, obs):
            return {"actions": np.zeros((1, 8), dtype=np.float32)}
        
        def reset(self, reset_info):
            pass
    
    logging.basicConfig(level=logging.INFO)
    policy = DummyPolicy()
    server = WebsocketPolicyServer(policy, PolicyServerConfig())
    server.serve_forever()
        
