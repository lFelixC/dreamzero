"""Client for communicating with a policy server.

Adapted from https://github.com/robo-arena/roboarena/

"""

import logging
import time
from typing import Any, Dict, Tuple

import websockets.sync.client
from typing_extensions import override

from eval_utils.base_policy import BasePolicy
from eval_utils import msgpack_numpy

# The policy server handles inference synchronously, so it may be unable to
# respond to keepalive pings during a long first forward / compile. Disable
# client pings by default; the blocking recv is the liveness signal we care
# about for eval.
OPEN_TIMEOUT_SECS = 0
PING_INTERVAL_SECS = None
PING_TIMEOUT_SECS = None


def _normalize_timeout(timeout_seconds: float | None) -> float | None:
    if timeout_seconds is None:
        return None
    if timeout_seconds <= 0:
        return None
    return timeout_seconds


class WebsocketClientPolicy(BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        log_wait: bool = True,
        open_timeout: float | None = OPEN_TIMEOUT_SECS,
    ) -> None:
        self._uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        self._log_wait = log_wait
        self._open_timeout = _normalize_timeout(open_timeout)
        self._ws, self._server_metadata = self._wait_for_server()
        self.last_timing: dict[str, float] = {}
        self.last_server_timing: dict[str, Any] = {}

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        if self._log_wait:
            logging.info(f"Waiting for server at {self._uri}...")
        try:
            conn = websockets.sync.client.connect(
                self._uri, 
                compression=None, 
                max_size=None,
                open_timeout=self._open_timeout,
                ping_interval=PING_INTERVAL_SECS,
                ping_timeout=PING_TIMEOUT_SECS,
            )
            metadata = msgpack_numpy.unpackb(conn.recv())
            return conn, metadata
        except Exception:
            if self._log_wait:
                logging.info("Connection to server with ws:// failed. Trying wss:// ...")
            
        self._uri = "wss://" + self._uri.split("//")[1]
        conn = websockets.sync.client.connect(
            self._uri, 
            compression=None, 
            max_size=None,
            open_timeout=self._open_timeout,
            ping_interval=PING_INTERVAL_SECS,
            ping_timeout=PING_TIMEOUT_SECS,
        )
        metadata = msgpack_numpy.unpackb(conn.recv())
        return conn, metadata

    def close(self) -> None:
        self._ws.close()

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        actions, _, _ = self.infer_timed(obs)
        return actions

    def infer_timed(self, obs: Dict) -> tuple[Any, dict[str, float], dict[str, Any]]:  # noqa: UP006
        # Notify server that we're calling the infer endpoint (as opposed to the reset endpoint)
        request = dict(obs)
        request["endpoint"] = "infer"

        request_start = time.perf_counter()
        pack_start = time.perf_counter()
        data = self._packer.pack(request)
        pack_obs_time = time.perf_counter() - pack_start
        send_start = time.perf_counter()
        self._ws.send(data)
        send_time = time.perf_counter() - send_start
        recv_start = time.perf_counter()
        response = self._ws.recv()
        recv_time = time.perf_counter() - recv_start
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        unpack_start = time.perf_counter()
        unpacked = msgpack_numpy.unpackb(response)
        unpack_time = time.perf_counter() - unpack_start
        total_time = time.perf_counter() - request_start

        timing = {
            "pack_obs_time": pack_obs_time,
            "websocket_send_time": send_time,
            "websocket_recv_time": recv_time,
            "unpack_action_time": unpack_time,
            "request_total_time": total_time,
            "T_send_obs": pack_obs_time + send_time,
            "T_recv_action": recv_time + unpack_time,
        }
        server_timing: dict[str, Any] = {}
        if isinstance(unpacked, dict) and "actions" in unpacked:
            raw_server_timing = unpacked.get("server_timing", {})
            if isinstance(raw_server_timing, dict):
                server_timing = raw_server_timing
            self.last_timing = timing
            self.last_server_timing = server_timing
            return unpacked["actions"], timing, server_timing
        self.last_timing = timing
        self.last_server_timing = server_timing
        return unpacked, timing, server_timing

    @override
    def reset(self, reset_info: Dict) -> None:
        response, _ = self.reset_timed(reset_info)
        return response

    def reset_timed(self, reset_info: Dict) -> tuple[Any, dict[str, float]]:  # noqa: UP006
        # Notify server that we're calling the reset endpoint (as opposed to the infer endpoint)
        request = dict(reset_info)
        request["endpoint"] = "reset"

        request_start = time.perf_counter()
        data = self._packer.pack(request)
        send_start = time.perf_counter()
        self._ws.send(data)
        send_time = time.perf_counter() - send_start
        recv_start = time.perf_counter()
        response = self._ws.recv()
        recv_time = time.perf_counter() - recv_start
        timing = {
            "websocket_send_time": send_time,
            "websocket_recv_time": recv_time,
            "request_total_time": time.perf_counter() - request_start,
        }
        return response, timing

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = WebsocketClientPolicy()
    actions = client.infer({})
    print(f"Actions received: {actions}")
    client.reset({})
