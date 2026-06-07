# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# This file is modified from https://github.com/haotian-liu/LLaVA/

from abc import ABC
from collections.abc import Mapping
import contextlib
import gc
import inspect
import json
import logging
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Optional
import warnings

from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader, Dataset, Sampler
import transformers
from transformers import TrainerCallback, set_seed
from transformers.trainer import (
    # ALL_LAYERNORM_LAYERS,  # ShardedDDPOption,  # Removed deprecated import
    TRAINER_STATE_NAME,
    TrainerState,
    get_last_checkpoint,
    get_parameter_names,
    is_sagemaker_mp_enabled,
)

import groot.vla.common.utils as U
from groot.vla.data.dataset.lerobot_sharded import ShardedLeRobotMixtureDataset
from groot.vla.data.schema import EmbodimentTag
from groot.vla.data.transform import ComposedModalityTransform
from groot.vla.experiment.utils import (
    compute_grad_accum_to_match_global_bs,
    dtype_from_string,
    get_checkpoint_path,
    mprint,
    safe_save_model_for_hf_trainer,
)
from groot.vla.utils.checkpoint_sidecar import (
    materialize_component_sidecars,
    prepare_action_head_cfg_for_checkpoint,
)
from groot.vla.utils.nvtx_utils import nvtx_enabled, nvtx_range
from groot.vla.utils.timer import ContextTimer

logger = logging.getLogger(__name__)

# Fix resume: https://github.com/huggingface/transformers/pull/34632/files
np_core = np.core
allowlist = [np_core.multiarray._reconstruct, np.ndarray, np.dtype]
# numpy >1.25 defines numpy.dtypes.UInt32DType, but below works for
# all versions of numpy
allowlist += [type(np.dtype(np.uint32))]
torch.serialization.add_safe_globals(allowlist)

# Define LayerNorm classes locally to replace deprecated ALL_LAYERNORM_LAYERS
LAYERNORM_LAYERS = [
    torch.nn.LayerNorm,
    torch.nn.GroupNorm,
    torch.nn.InstanceNorm1d,
    torch.nn.InstanceNorm2d,
    torch.nn.InstanceNorm3d,
    torch.nn.LocalResponseNorm,
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
]


class LossLoggerCallback(TrainerCallback):
    """Callback that writes per-step loss metrics to a JSONL file for offline analysis."""

    def __init__(self, output_path: str):
        self.output_path = output_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or logs is None:
            return
        entry = {"step": state.global_step}
        for key in (
            "loss",
            "dynamics_loss_avg",
            "action_loss_avg",
            "dynamics_loss_contribution_avg",
            "action_loss_contribution_avg",
            "train/loss_total",
            "train/loss_video",
            "train/loss_action",
            "train/loss_video_contribution",
            "train/loss_action_contribution",
            "train/loss_weight_dynamics",
            "train/loss_weight_action",
            "val/loss_total",
            "val/loss_video",
            "val/loss_action",
            "val/loss_video_contribution",
            "val/loss_action_contribution",
            "learning_rate",
        ):
            if key in logs:
                entry[key] = logs[key]
        for key, value in logs.items():
            if (
                key.startswith("train/mot_")
                or key.startswith("mot_")
                or key.startswith("train/local_grad/")
                or key.startswith("local_grad/")
            ):
                entry[key] = value
        if len(entry) > 1:  # more than just "step"
            with open(self.output_path, "a") as f:
                f.write(json.dumps(entry) + "\n")


class NVTXTrainerCallback(TrainerCallback):
    """Adds light NVTX ranges around optimizer.step without touching Trainer internals."""

    def __init__(self):
        self._optimizer_step_active = False
        self._optimizer_step_start = None

    @staticmethod
    def _timing_enabled() -> bool:
        return os.environ.get("DREAMZERO_TIMING_DEBUG", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    @staticmethod
    def _sync_cuda_for_timing() -> None:
        if os.environ.get("DREAMZERO_TIMING_SYNC_CUDA", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        } and torch.cuda.is_available():
            torch.cuda.synchronize()

    @classmethod
    def _write_timing_event(cls, args, state, event: str, **fields) -> None:
        if not cls._timing_enabled():
            return
        output_dir = Path(args.output_dir)
        timing_dir = output_dir / "timing"
        timing_dir.mkdir(parents=True, exist_ok=True)
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        entry = {
            "event": event,
            "time": time.time(),
            "rank": rank,
            "local_rank": local_rank,
            "global_step": int(getattr(state, "global_step", -1)),
        }
        entry.update(fields)
        with open(timing_dir / f"timing_rank{rank}_local{local_rank}.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if self._timing_enabled():
            self._sync_cuda_for_timing()
            self._optimizer_step_start = time.perf_counter()
        if not nvtx_enabled():
            return control
        torch.cuda.nvtx.range_push("dreamzero.train.optimizer_step")
        self._optimizer_step_active = True
        return control

    def on_optimizer_step(self, args, state, control, **kwargs):
        if self._optimizer_step_active:
            torch.cuda.nvtx.range_pop()
            self._optimizer_step_active = False
        if self._optimizer_step_start is not None:
            self._sync_cuda_for_timing()
            elapsed = time.perf_counter() - self._optimizer_step_start
            self._optimizer_step_start = None
            self._write_timing_event(args, state, "optimizer_step", seconds=elapsed)
        return control


def maybe_enable_swanlab_sync():
    """Optionally route wandb logs through SwanLab with minimal intrusion.

    Enable by setting `SWANLAB_SYNC_WANDB=1` in the environment. If `swanlab`
    is not installed, training continues with plain wandb logging.
    """
    if os.environ.get("SWANLAB_SYNC_WANDB", "").lower() not in {"1", "true", "yes", "on"}:
        return

    # Only initialize the bridge on the main process.
    if os.environ.get("RANK", "0") != "0":
        return

    try:
        import swanlab
    except ImportError:
        print("SWANLAB_SYNC_WANDB is enabled, but `swanlab` is not installed. Falling back to plain wandb.")
        return

    swanlab.sync_wandb()
    print("Enabled SwanLab sync for wandb logging.")


class CheckpointFormatCallback(TrainerCallback):
    """This callback format checkpoint to make them standalone. For now, it copies all config
    files to /checkpoint-{step}/experiment_cfg/:
    - conf.yaml
    - initial_actions.npz
    - metadata.json
    """

    def __init__(
        self, run_name: str, exp_cfg_dir: Path | None = None, processor_dir: Path | None = None
    ):
        """
        Args:
            run_name: Name of the experiment run
            exp_cfg_dir: Path to the directory containing all experiment metadata
        """
        self.exp_cfg_dir = exp_cfg_dir
        self.processor_dir = processor_dir

    def on_save(self, args, state, control, **kwargs):
        """Called after the trainer saves a checkpoint."""
        if state.is_world_process_zero:
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"

            # Copy experiment config directory if provided
            if self.exp_cfg_dir is not None:
                exp_cfg_dst = checkpoint_dir / self.exp_cfg_dir.name
                if self.exp_cfg_dir.exists():
                    print(
                        f"Copying experiment config directory {self.exp_cfg_dir} to {exp_cfg_dst}"
                    )
                    shutil.copytree(self.exp_cfg_dir, exp_cfg_dst, dirs_exist_ok=True)

            # Copy processor directory if provided
            if self.processor_dir is not None:
                if self.processor_dir.exists():
                    print(f"Copying processor directory {self.processor_dir} to {checkpoint_dir}")
                    shutil.copytree(self.processor_dir, checkpoint_dir, dirs_exist_ok=True)

            # Copy wandb_config.json if provided
            wandb_config_src = Path(args.output_dir) / "wandb_config.json"
            wandb_config_dst = checkpoint_dir / "wandb_config.json"
            if wandb_config_src.exists():
                print(f"Copying wandb_config.json from {wandb_config_src} to {wandb_config_dst}")
                shutil.copy2(wandb_config_src, wandb_config_dst)


class ProfCallback(transformers.TrainerCallback):
    """Callback to manage PyTorch profiler during training.

    Dynamically starts/stops the profiler within a specified session step window.
    After profiling completes, triggers optional S3 upload and removes itself.

    Args:
        profile_dir: Directory to save profile traces
        upload_callback: Optional callback to trigger S3 upload after profiling
        profile_start_step: Session step to start profiling (default: 50)
        profile_end_step: Session step to stop profiling
        warmup_steps: Number of warmup steps for profiler schedule (default: 1)
        active_steps: Number of active profiling steps (default: 5)
        trainer: Trainer instance (required for self-removal after profiling)
        record_shapes: Record tensor shapes in profiler (default: False)
        with_stack: Record Python stack traces (default: True)
        profile_memory: Record memory allocation/deallocation (default: False)
    """

    def __init__(
        self,
        profile_dir,
        upload_callback=None,
        profile_start_step=50,
        profile_end_step=55,
        warmup_steps=1,
        active_steps=5,
        trainer=None,
        record_shapes=False,
        with_stack=True,
        profile_memory=False,
    ):
        self.profile_dir = profile_dir
        self.upload_callback = upload_callback
        self.profile_start_step = profile_start_step
        self.profile_end_step = profile_end_step
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps
        self.trainer = trainer
        self.record_shapes = record_shapes
        self.with_stack = with_stack
        self.profile_memory = profile_memory
        self.upload_triggered = False
        self.starting_global_step = None
        self.session_step = 0
        self.prof = None
        self.profiling_active = False
        self.profiling_complete = False
        self.removed_from_trainer = False

    def on_step_begin(self, args, state, control, **kwargs):
        # Remove callback after upload triggered to eliminate all overhead
        if self.profiling_complete and self.upload_triggered and not self.removed_from_trainer:
            if self.trainer is not None and hasattr(self.trainer, "callback_handler"):
                try:
                    self.trainer.callback_handler.callbacks.remove(self)
                    self.removed_from_trainer = True
                    logging.info(
                        f"Removed ProfCallback from trainer at global step {state.global_step}"
                    )
                except (ValueError, AttributeError) as e:
                    logging.warning(f"Failed to remove ProfCallback: {e}")
            return

        # Early return if profiling already complete
        if self.profiling_complete:
            return

        # Record starting global step on first call
        if self.starting_global_step is None:
            self.starting_global_step = state.global_step

        # Calculate session step
        self.session_step = state.global_step - self.starting_global_step

        # Start profiler when we reach the profiling window
        if self.session_step == self.profile_start_step and self.prof is None:
            logging.info(
                f"Starting profiler at global step {state.global_step} (session step {self.session_step})"
            )
            self.prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    skip_first=0,
                    wait=0,
                    warmup=self.warmup_steps,
                    active=self.active_steps,
                    repeat=1,
                ),
                profile_memory=self.profile_memory,
                with_stack=self.with_stack,
                record_shapes=self.record_shapes,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(self.profile_dir)),
            )
            self.prof.__enter__()
            self.profiling_active = True

    def on_step_end(self, args, state, control, **kwargs):
        # Early return if profiling already complete
        if self.profiling_complete:
            return

        # Recalculate session_step to ensure accuracy
        if self.starting_global_step is not None:
            self.session_step = state.global_step - self.starting_global_step

        # Step profiler if active
        if self.profiling_active and self.prof is not None:
            self.prof.step()

        # Stop profiler when we reach the end of profiling window
        if self.session_step == self.profile_end_step and self.prof is not None:
            self.prof.__exit__(None, None, None)
            self.profiling_active = False

            # Explicitly release profiler resources to minimize CUPTI overhead
            # Combined with TEARDOWN_CUPTI=1 env var for full cleanup
            del self.prof
            self.prof = None

            # Force CUDA synchronization to ensure profiler cleanup completes
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            self.profiling_complete = True
            logging.info(
                f"Profiler stopped and resources released at global step {state.global_step} "
                f"(session step {self.session_step})"
            )

            # Trigger upload if callback provided
            if self.upload_callback:
                logging.info(f"Triggering upload at global step {state.global_step}...")
                self.upload_callback()

            # Mark as ready for callback removal
            self.upload_triggered = True


class BaseSampler(Sampler):
    """Sampler for dataset, which enables `set_epoch` for Dataset.
    `set_epoch` will be called by huggingface Trainer at the end of each epoch.
    `shuffle` is also supported for training set shuffling
    """

    def __init__(self, data_source: Dataset, shuffle: bool = False, seed: int = 0):
        self.data_source = data_source
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # must not add rank here, or randomization will be different for each rank
            return iter(torch.randperm(len(self.data_source), generator=g).tolist())
        return iter(range(len(self.data_source)))

    def set_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self.data_source, "set_epoch"):
            # this is important for dataset
            self.data_source.set_epoch(epoch)

    def __len__(self):
        return len(self.data_source)


class BaseTrainer(transformers.Trainer):
    BACKBONE_GRAD_CLIP_NORM = 1.0

    def __init__(self, **kwargs):
        # Increase the cache size limit for torch._dynamo to
        # accommodate videos with different numbers of frames.
        torch._dynamo.config.cache_size_limit = 1000

        self.compute_dtype = kwargs.pop("compute_dtype")
        self.output_dir = kwargs.pop("output_dir")
        self.timer = ContextTimer(self)

        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.global_rank = int(os.environ.get("RANK", "0"))
        self.node_rank = int(os.environ.get("NODE_RANK", "0"))
        self.timing_debug = os.environ.get("DREAMZERO_TIMING_DEBUG", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.timing_sync_cuda = os.environ.get(
            "DREAMZERO_TIMING_SYNC_CUDA", "1" if self.timing_debug else "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._timing_log_path = None
        self._timing_lock = threading.Lock()
        self._last_batch_fetch_seconds = None
        self._last_prepare_inputs_seconds = None
        self._last_forward_seconds = None
        self._last_loss_logging_seconds = None
        self._last_backward_seconds = None
        legacy_track_loss_grad = kwargs.pop("track_loss_grad_conflict", None)
        self.track_loss_grad = self._coerce_bool(
            kwargs.pop(
                "track_loss_grad",
                legacy_track_loss_grad if legacy_track_loss_grad is not None else False,
            )
        )
        if "DREAMZERO_TRACK_LOSS_GRAD" in os.environ:
            self.track_loss_grad = self._coerce_bool(os.environ["DREAMZERO_TRACK_LOSS_GRAD"])
        self.loss_grad_logging_steps = self._coerce_positive_int(
            kwargs.pop(
                "loss_grad_logging_steps",
                kwargs.pop("grad_conflict_logging_steps", 10),
            ),
            default=10,
        )
        if "DREAMZERO_LOSS_GRAD_LOGGING_STEPS" in os.environ:
            self.loss_grad_logging_steps = self._coerce_positive_int(
                os.environ["DREAMZERO_LOSS_GRAD_LOGGING_STEPS"],
                default=self.loss_grad_logging_steps,
            )
        self.action_expert_weight_decay = self._coerce_optional_nonnegative_float(
            kwargs.pop("action_expert_weight_decay", None),
            name="action_expert_weight_decay",
        )
        self.dataloader_in_order = self._coerce_bool(
            kwargs.pop("dataloader_in_order", True)
        )
        if "DREAMZERO_DATALOADER_IN_ORDER" in os.environ:
            self.dataloader_in_order = self._coerce_bool(
                os.environ["DREAMZERO_DATALOADER_IN_ORDER"]
            )
        self._loss_grad_disabled_warning_printed = False
        if self.timing_debug:
            timing_dir = Path(self.output_dir) / "timing"
            timing_dir.mkdir(parents=True, exist_ok=True)
            self._timing_log_path = (
                timing_dir / f"timing_rank{self.global_rank}_local{self.local_rank}.jsonl"
            )

        # Get distributed info
        self.current_step = 0

        # Profiling (legacy per-step profiling)
        self.enable_profiling = kwargs.pop("enable_profiling", False)
        self.profiling_steps = kwargs.pop("profiling_steps", 5)
        # Pop new ProfCallback config options (handled in create_trainer, not here)
        kwargs.pop("enable_prof_callback", None)
        kwargs.pop("profile_start_step", None)
        kwargs.pop("profile_warmup_steps", None)
        kwargs.pop("profile_active_steps", None)
        kwargs.pop("profile_record_shapes", None)
        kwargs.pop("profile_with_stack", None)
        kwargs.pop("profile_memory", None)
        kwargs.pop("msc_profile_url", None)
        kwargs.pop("profile_delete_after_upload", None)
        if self.enable_profiling:
            # Setup profiling directories
            self.profile_dir = Path(self.output_dir) / "profiling"
            self.memory_profile_dir = self.profile_dir / "memory"
            self.torch_profile_dir = self.profile_dir / "torch"

            self.memory_profile_dir.mkdir(exist_ok=True, parents=True)
            self.torch_profile_dir.mkdir(exist_ok=True, parents=True)

            # Start recording the memory history.
            torch.cuda.memory._record_memory_history(max_entries=100000)

        super().__init__(**kwargs)
        self._install_backward_nvtx_wrapper()
        self._install_backbone_grad_clip_wrapper()

        self.loss_queues = {}
        self.loss_queue_size = 10
        self._eval_loss_sums = {}
        self._eval_loss_count = 0
        self._backbone_grad_clip_logged = False

    @staticmethod
    def _loss_aliases(outputs) -> dict[str, float]:
        aliases = {
            "loss_total": outputs.get("loss"),
            "loss_video": outputs.get("dynamics_loss"),
            "loss_action": outputs.get("action_loss"),
            "loss_video_contribution": outputs.get("dynamics_loss_contribution"),
            "loss_action_contribution": outputs.get("action_loss_contribution"),
        }
        result = {}
        for key, value in aliases.items():
            if value is None:
                continue
            result[key] = value.detach().float().mean().item() if torch.is_tensor(value) else float(value)
        return result

    @staticmethod
    def _metric_aliases(outputs) -> dict[str, float]:
        result = {}
        for key, value in outputs.items():
            if not (key.startswith("mot_") or key.startswith("loss_weight_")):
                continue
            if value is None:
                continue
            result[key] = value.detach().float().mean().item() if torch.is_tensor(value) else float(value)
        return result

    @staticmethod
    def _coerce_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @staticmethod
    def _coerce_positive_int(value, *, default: int) -> int:
        try:
            value = int(value)
        except (TypeError, ValueError):
            return default
        return max(value, 1)

    @staticmethod
    def _coerce_optional_nonnegative_float(value, *, name: str) -> float | None:
        if value is None or value == "null":
            return None
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a non-negative float or null, got {value!r}") from exc
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative, got {value}")
        return value

    @staticmethod
    def _is_mot_action_branch_param(name: str) -> bool:
        return name.startswith("action_expert.") or ".action_expert." in name

    @staticmethod
    def _is_action_head_diffusion_param(name: str) -> bool:
        return name.startswith("action_head.model.") or ".action_head.model." in name

    def _is_backbone_grad_clip_param(self, name: str) -> bool:
        # In DreamZero MoT, the trainable "backbone" is the video diffusion branch.
        if self._is_action_head_diffusion_param(name):
            return not self._is_mot_action_branch_param(name)
        return name.startswith("backbone.") or ".backbone." in name

    @staticmethod
    def _grad_norm_sq(grads, *, device: torch.device) -> torch.Tensor:
        norm_sq = torch.zeros((), device=device, dtype=torch.float32)
        for grad in grads:
            if grad is None:
                continue
            grad = grad.detach()
            norm_sq = norm_sq + grad.float().pow(2).sum()
        return norm_sq

    @staticmethod
    def _grad_dot(grads_a, grads_b, *, device: torch.device) -> torch.Tensor:
        dot = torch.zeros((), device=device, dtype=torch.float32)
        for grad_a, grad_b in zip(grads_a, grads_b):
            if grad_a is None or grad_b is None:
                continue
            dot = dot + (grad_a.detach().float() * grad_b.detach().float()).sum()
        return dot

    @staticmethod
    def _empty_grads(length: int):
        return (None,) * length

    def _loss_grads(self, loss: torch.Tensor, params: list[torch.nn.Parameter]):
        if not params or not torch.is_tensor(loss) or not loss.requires_grad:
            return self._empty_grads(len(params))
        return torch.autograd.grad(
            loss,
            params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )

    @staticmethod
    def _average_grad_stat_scalars(scalars: list[torch.Tensor]) -> list[torch.Tensor]:
        if not scalars:
            return []
        stacked = torch.stack([scalar.detach().float() for scalar in scalars])
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # These are rank-local grad statistics averaged for logging, not the
            # norm/cosine of the final optimizer gradient after DDP/DeepSpeed reduction.
            torch.distributed.all_reduce(stacked, op=torch.distributed.ReduceOp.SUM)
            stacked = stacked / torch.distributed.get_world_size()
        return list(stacked.unbind())

    @staticmethod
    def _grad_norm_and_contrib(norm_sq: torch.Tensor, loss_weight: float) -> tuple[float, float]:
        norm = norm_sq.sqrt().item()
        return norm, abs(loss_weight) * norm

    @staticmethod
    def _grad_cosine(
        dot: torch.Tensor,
        norm_a_sq: torch.Tensor,
        norm_b_sq: torch.Tensor,
    ) -> float:
        norm_a = torch.clamp(norm_a_sq, min=0.0).sqrt()
        norm_b = torch.clamp(norm_b_sq, min=0.0).sqrt()
        denom = norm_a * norm_b
        if denom.item() == 0.0:
            return float("nan")
        return torch.clamp(dot / denom, min=-1.0, max=1.0).item()

    def _grad_metric_root_model(self, model):
        try:
            return self.accelerator.unwrap_model(model)
        except Exception:
            return model

    @staticmethod
    def _is_mot_architecture(root_model) -> bool:
        action_head = getattr(root_model, "action_head", None)
        if action_head is None:
            return False
        head_config = getattr(action_head, "config", None)
        if getattr(head_config, "architecture", "joint") == "mot":
            return True
        head_model = getattr(action_head, "model", None)
        return bool(getattr(head_model, "is_mot_wam", False))

    def _trainable_named_parameters_for_grad_metrics(self, model):
        root_model = self._grad_metric_root_model(model)
        named_params = [
            (name, param)
            for name, param in root_model.named_parameters()
            if param.requires_grad
        ]
        return root_model, named_params

    def _mot_grad_parameter_groups(
        self,
        named_params: list[tuple[str, torch.nn.Parameter]],
    ) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
        action_params = [
            param for name, param in named_params if self._is_mot_action_branch_param(name)
        ]
        video_params = [
            param
            for name, param in named_params
            if self._is_action_head_diffusion_param(name)
            and not self._is_mot_action_branch_param(name)
        ]
        if not video_params:
            video_params = [
                param
                for name, param in named_params
                if not self._is_mot_action_branch_param(name)
            ]
        return video_params, action_params

    @staticmethod
    def _first_metric_device(
        losses: list[torch.Tensor],
        params: list[torch.nn.Parameter],
    ) -> torch.device:
        for loss in losses:
            if torch.is_tensor(loss):
                return loss.device
        for param in params:
            return param.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _joint_loss_grad_metrics(
        self,
        dynamics_loss: torch.Tensor,
        action_loss: torch.Tensor,
        named_params: list[tuple[str, torch.nn.Parameter]],
        dynamics_loss_weight: float = 1.0,
        action_loss_weight: float = 1.0,
    ) -> dict[str, float]:
        params = [param for _, param in named_params]
        device = self._first_metric_device([dynamics_loss, action_loss], params)
        dynamics_grads = self._loss_grads(dynamics_loss, params)
        action_grads = self._loss_grads(action_loss, params)

        dynamics_norm_sq = self._grad_norm_sq(dynamics_grads, device=device)
        action_norm_sq = self._grad_norm_sq(action_grads, device=device)
        dot = self._grad_dot(dynamics_grads, action_grads, device=device)
        del dynamics_grads, action_grads

        dynamics_norm_sq, action_norm_sq, dot = self._average_grad_stat_scalars(
            [dynamics_norm_sq, action_norm_sq, dot]
        )
        cosine = self._grad_cosine(dot, dynamics_norm_sq, action_norm_sq)
        dynamics_norm, dynamics_contrib_norm = self._grad_norm_and_contrib(
            dynamics_norm_sq, dynamics_loss_weight
        )
        action_norm, action_contrib_norm = self._grad_norm_and_contrib(
            action_norm_sq, action_loss_weight
        )
        return {
            "local_grad/joint/dynamics_grad_norm": dynamics_norm,
            "local_grad/joint/action_grad_norm": action_norm,
            "local_grad/joint/dynamics_contrib_grad_norm": dynamics_contrib_norm,
            "local_grad/joint/action_contrib_grad_norm": action_contrib_norm,
            "local_grad/joint/dynamics_action_cosine": cosine,
        }

    def _mot_loss_grad_metrics(
        self,
        dynamics_loss: torch.Tensor,
        action_loss: torch.Tensor,
        named_params: list[tuple[str, torch.nn.Parameter]],
        dynamics_loss_weight: float = 1.0,
        action_loss_weight: float = 1.0,
    ) -> dict[str, float]:
        video_params, action_params = self._mot_grad_parameter_groups(named_params)
        all_action_loss_params = video_params + action_params
        device = self._first_metric_device(
            [dynamics_loss, action_loss],
            all_action_loss_params or video_params,
        )

        dynamics_video_grads = self._loss_grads(dynamics_loss, video_params)
        action_loss_grads = self._loss_grads(action_loss, all_action_loss_params)
        action_video_grads = action_loss_grads[: len(video_params)]
        action_action_grads = action_loss_grads[len(video_params) :]

        dynamics_video_norm_sq = self._grad_norm_sq(dynamics_video_grads, device=device)
        action_video_norm_sq = self._grad_norm_sq(action_video_grads, device=device)
        action_action_norm_sq = self._grad_norm_sq(action_action_grads, device=device)
        video_dot = self._grad_dot(dynamics_video_grads, action_video_grads, device=device)
        del dynamics_video_grads, action_loss_grads, action_video_grads, action_action_grads

        (
            dynamics_video_norm_sq,
            action_video_norm_sq,
            action_action_norm_sq,
            video_dot,
        ) = self._average_grad_stat_scalars(
            [
                dynamics_video_norm_sq,
                action_video_norm_sq,
                action_action_norm_sq,
                video_dot,
            ]
        )
        cosine = self._grad_cosine(
            video_dot,
            dynamics_video_norm_sq,
            action_video_norm_sq,
        )
        dynamics_video_norm, dynamics_video_contrib_norm = self._grad_norm_and_contrib(
            dynamics_video_norm_sq, dynamics_loss_weight
        )
        action_video_norm, action_video_contrib_norm = self._grad_norm_and_contrib(
            action_video_norm_sq, action_loss_weight
        )
        action_action_norm, action_action_contrib_norm = self._grad_norm_and_contrib(
            action_action_norm_sq, action_loss_weight
        )
        return {
            "local_grad/mot/dynamics_video_grad_norm": dynamics_video_norm,
            "local_grad/mot/action_video_grad_norm": action_video_norm,
            "local_grad/mot/action_action_grad_norm": action_action_norm,
            "local_grad/mot/dynamics_video_contrib_grad_norm": dynamics_video_contrib_norm,
            "local_grad/mot/action_video_contrib_grad_norm": action_video_contrib_norm,
            "local_grad/mot/action_action_contrib_grad_norm": action_action_contrib_norm,
            "local_grad/mot/video_dynamics_action_cosine": cosine,
        }

    @staticmethod
    def _output_scalar(outputs, key: str, default: float) -> float:
        value = outputs.get(key)
        if value is None:
            return default
        if torch.is_tensor(value):
            return value.detach().float().mean().item()
        return float(value)

    def _maybe_log_loss_grad_metrics(self, model, outputs) -> None:
        if not self.track_loss_grad:
            return
        if self.current_step % self.loss_grad_logging_steps != 0:
            return
        dynamics_loss = outputs.get("dynamics_loss")
        action_loss = outputs.get("action_loss")
        if not torch.is_tensor(dynamics_loss) or not torch.is_tensor(action_loss):
            return

        try:
            dynamics_loss_weight = self._output_scalar(outputs, "loss_weight_dynamics", 1.0)
            action_loss_weight = self._output_scalar(outputs, "loss_weight_action", 1.0)
            root_model, named_params = self._trainable_named_parameters_for_grad_metrics(model)
            if not named_params:
                return
            if self._is_mot_architecture(root_model):
                metrics = self._mot_loss_grad_metrics(
                    dynamics_loss,
                    action_loss,
                    named_params,
                    dynamics_loss_weight=dynamics_loss_weight,
                    action_loss_weight=action_loss_weight,
                )
            else:
                metrics = self._joint_loss_grad_metrics(
                    dynamics_loss,
                    action_loss,
                    named_params,
                    dynamics_loss_weight=dynamics_loss_weight,
                    action_loss_weight=action_loss_weight,
                )
        except RuntimeError:
            if not self._loss_grad_disabled_warning_printed:
                logger.exception("Disabling DreamZero loss-gradient metrics after failure.")
                self._loss_grad_disabled_warning_printed = True
            self.track_loss_grad = False
            return

        self.log({f"train/{key}": value for key, value in metrics.items()})

    def _install_backward_nvtx_wrapper(self) -> None:
        original_backward = self.accelerator.backward
        if getattr(original_backward, "_dreamzero_nvtx_wrapped", False):
            return

        def wrapped_backward(loss, **kwargs):
            start = self._timing_start() if self.timing_debug else None
            with nvtx_range("dreamzero.train.backward"):
                result = original_backward(loss, **kwargs)
            if start is not None:
                self._last_backward_seconds = self._timing_elapsed(start)
            return result

        wrapped_backward._dreamzero_nvtx_wrapped = True
        self.accelerator.backward = wrapped_backward

    def _backbone_grad_clip_parameters(self) -> list[torch.nn.Parameter]:
        model = getattr(self, "model_wrapped", None) or self.model
        root_model = self._grad_metric_root_model(model)
        return [
            param
            for name, param in root_model.named_parameters()
            if (
                param.requires_grad
                and param.grad is not None
                and self._is_backbone_grad_clip_param(name)
            )
        ]

    def _clip_backbone_grad_norm(self) -> None:
        params = self._backbone_grad_clip_parameters()
        if not params:
            return
        if self.global_rank == 0 and not self._backbone_grad_clip_logged:
            print(
                "Clipping DreamZero backbone gradients with "
                f"max_norm={self.BACKBONE_GRAD_CLIP_NORM}"
            )
            self._backbone_grad_clip_logged = True
        torch.nn.utils.clip_grad_norm_(
            params,
            max_norm=self.BACKBONE_GRAD_CLIP_NORM,
            norm_type=2,
        )

    def _install_backbone_grad_clip_wrapper(self) -> None:
        original_on_pre_optimizer_step = self.callback_handler.on_pre_optimizer_step
        if getattr(original_on_pre_optimizer_step, "_dreamzero_backbone_clip_wrapped", False):
            return

        def wrapped_on_pre_optimizer_step(args, state, control, **kwargs):
            self._clip_backbone_grad_norm()
            return original_on_pre_optimizer_step(args, state, control, **kwargs)

        wrapped_on_pre_optimizer_step._dreamzero_backbone_clip_wrapped = True
        self.callback_handler.on_pre_optimizer_step = wrapped_on_pre_optimizer_step

    def _timing_sync(self) -> None:
        if self.timing_debug and self.timing_sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _timing_start(self) -> float:
        self._timing_sync()
        return time.perf_counter()

    def _timing_elapsed(self, start: float) -> float:
        self._timing_sync()
        return time.perf_counter() - start

    def _write_timing_event(self, event: str, **fields) -> None:
        if not self.timing_debug or self._timing_log_path is None:
            return
        entry = {
            "event": event,
            "time": time.time(),
            "rank": self.global_rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "global_step": int(getattr(self.state, "global_step", -1)),
            "current_step": int(getattr(self, "current_step", -1)),
        }
        entry.update(fields)
        with self._timing_lock:
            with open(self._timing_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        start = time.perf_counter()
        with nvtx_range("dreamzero.train.dataloader_wait"):
            result = super().get_batch_samples(epoch_iterator, num_batches, device)
        elapsed = time.perf_counter() - start
        self._last_batch_fetch_seconds = elapsed
        self._write_timing_event("batch_fetch", seconds=elapsed, num_batches=num_batches)
        return result

    @staticmethod
    def _nvtx_input_path(parent: Optional[str], child: object) -> str:
        child_path = str(child)
        return f"{parent}.{child_path}" if parent else child_path

    @staticmethod
    def _nvtx_tensor_label(path: Optional[str], tensor: torch.Tensor) -> str:
        name = path or "tensor"
        shape = "x".join(str(dim) for dim in tensor.shape) or "scalar"
        dtype = str(tensor.dtype).replace("torch.", "")
        return f"{name}[shape={shape},dtype={dtype},src={tensor.device}]"

    def _prepare_input(self, data, nvtx_path: Optional[str] = None):
        if not nvtx_enabled() and nvtx_path is None:
            return super()._prepare_input(data)

        if isinstance(data, Mapping):
            return type(data)(
                {
                    key: self._prepare_input(value, self._nvtx_input_path(nvtx_path, key))
                    for key, value in data.items()
                }
            )
        if isinstance(data, (tuple, list)):
            return type(data)(
                self._prepare_input(value, self._nvtx_input_path(nvtx_path, index))
                for index, value in enumerate(data)
            )
        if isinstance(data, torch.Tensor):
            target_device = torch.device(self.args.device)
            transfer_kind = (
                "h2d" if data.device.type == "cpu" and target_device.type == "cuda" else "to_device"
            )
            label = self._nvtx_tensor_label(nvtx_path, data)
            with nvtx_range(f"dreamzero.train.{transfer_kind}.{label}->dst={target_device}"):
                kwargs = {"device": self.args.device}
                if self.is_deepspeed_enabled and (torch.is_floating_point(data) or torch.is_complex(data)):
                    kwargs.update(
                        {"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()}
                    )
                return data.to(**kwargs)
        return data

    def _prepare_inputs(self, inputs):
        if not nvtx_enabled():
            if not self.timing_debug:
                return super()._prepare_inputs(inputs)
            start = self._timing_start()
            inputs = super()._prepare_inputs(inputs)
            self._last_prepare_inputs_seconds = self._timing_elapsed(start)
            return inputs

        start = self._timing_start() if self.timing_debug else None
        with nvtx_range("dreamzero.train.prepare_inputs"):
            inputs = self._prepare_input(inputs, "batch")
            if len(inputs) == 0:
                raise ValueError(
                    "The batch received was empty, your model won't be able to train on it. "
                    f"Double-check that your training dataset contains keys expected by the model: "
                    f"{','.join(self._signature_columns)}."
                )
            if self.args.past_index >= 0 and self._past is not None:
                inputs["mems"] = self._past
            if start is not None:
                self._last_prepare_inputs_seconds = self._timing_elapsed(start)
            return inputs

    def _get_train_sampler(self):
        return BaseSampler(self.train_dataset, shuffle=True, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset):
        return BaseSampler(eval_dataset, shuffle=False)

    def training_step(self, model, inputs, num_items_in_batch=None):
        enable_profile = self.enable_profiling and self.current_step % self.profiling_steps == 0
        if enable_profile:
            profile_context = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=True,
            )
        else:
            profile_context = contextlib.nullcontext()

        self._last_prepare_inputs_seconds = None
        self._last_forward_seconds = None
        self._last_loss_logging_seconds = None
        self._last_backward_seconds = None
        start_time = self._timing_start() if self.timing_debug else time.time()
        with self.timer.with_label("training_step"), nvtx_range("dreamzero.train.training_step"), profile_context as prof:
            output = super().training_step(model, inputs)

        time_taken = (
            self._timing_elapsed(start_time) if self.timing_debug else time.time() - start_time
        )
        if self.timing_debug:
            accounted = sum(
                value
                for value in (
                    self._last_prepare_inputs_seconds,
                    self._last_forward_seconds,
                    self._last_loss_logging_seconds,
                    self._last_backward_seconds,
                )
                if value is not None
            )
            self._write_timing_event(
                "training_step",
                seconds=time_taken,
                batch_fetch_seconds=self._last_batch_fetch_seconds,
                prepare_inputs_seconds=self._last_prepare_inputs_seconds,
                forward_seconds=self._last_forward_seconds,
                loss_logging_seconds=self._last_loss_logging_seconds,
                backward_seconds=self._last_backward_seconds,
                unaccounted_seconds=max(time_taken - accounted, 0.0),
            )
        # print(
        #     f"Rank {self.global_rank} time taken for training_step {self.current_step}: {time_taken:.2f} seconds"
        # )

        if enable_profile:
            trace_path = f"{self.torch_profile_dir}/trace_rank_{self.global_rank}_step_{self.current_step}.json.gz"
            print(f"Rank {self.global_rank} exporting torch profile to {trace_path}")
            prof.export_chrome_trace(trace_path)

            snapshot_path = f"{self.memory_profile_dir}/memory_snapshot_rank_{self.global_rank}_step_{self.current_step}.pickle"
            print(f"Rank {self.global_rank} dumping memory snapshot to {snapshot_path}")
            torch.cuda.memory._dump_snapshot(snapshot_path)

        self.current_step += 1
        return output

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        forward_start = self._timing_start() if self.timing_debug else None
        with self.timer.with_label("model_forward"), nvtx_range("dreamzero.train.model_forward"):
            outputs = model(inputs)
            if forward_start is not None:
                self._last_forward_seconds = self._timing_elapsed(forward_start)

        loss_log_start = self._timing_start() if self.timing_debug else None
        loss_aliases = self._loss_aliases(outputs)
        metric_aliases = self._metric_aliases(outputs)
        if model.training:
            if self.current_step % self.loss_queue_size == 0 and loss_aliases:
                self.log({f"train/{key}": value for key, value in loss_aliases.items()})
            if self.current_step % self.loss_queue_size == 0 and metric_aliases:
                self.log({f"train/{key}": value for key, value in metric_aliases.items()})
        else:
            for key, value in loss_aliases.items():
                self._eval_loss_sums[key] = self._eval_loss_sums.get(key, 0.0) + value
            self._eval_loss_count += 1

        if model.training:
            ### For additional losses, track and log their moving averages
            for key, value in outputs.items():
                if (
                    (key.endswith("_loss") or key.endswith("_loss_contribution"))
                    and key != "loss"
                ):
                    # Initialize queue if not exists
                    if key not in self.loss_queues:
                        self.loss_queues[key] = []

                    # Add current loss value to queue
                    current_value = value.item() if torch.is_tensor(value) else value
                    self.loss_queues[key].append(current_value)

                    # Keep only last N values
                    if len(self.loss_queues[key]) > self.loss_queue_size:
                        self.loss_queues[key].pop(0)

                    # Log average every 10 steps
                    if self.current_step % self.loss_queue_size == 0:
                        avg_loss = sum(self.loss_queues[key]) / len(self.loss_queues[key])
                        self.log({f"{key}_avg": avg_loss})
        if loss_log_start is not None:
            self._last_loss_logging_seconds = self._timing_elapsed(loss_log_start)

        loss = outputs["loss"]
        if model.training:
            self._maybe_log_loss_grad_metrics(model, outputs)

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # VLA.forward accepts one batch dictionary, while HF Trainer's default
        # eval path calls model(**inputs) when there are no label fields. Keep
        # validation on the same loss/forward path as training.
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, return_outputs=False)
            loss = loss.detach().mean()
        return loss, None, None

    def evaluate(self, *args, **kwargs):
        self._eval_loss_sums = {}
        self._eval_loss_count = 0
        metrics = super().evaluate(*args, **kwargs)
        if self._eval_loss_count > 0:
            logs = {
                f"val/{key}": value / self._eval_loss_count
                for key, value in self._eval_loss_sums.items()
            }
            self.log(logs)
            metrics.update(logs)
        return metrics

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            action_expert_weight_decay = self.action_expert_weight_decay
            named_parameters = [
                (name, param)
                for name, param in opt_model.named_parameters()
                if param.requires_grad
            ]
            if action_expert_weight_decay is None:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in named_parameters
                            if n in decay_parameters
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in named_parameters
                            if n not in decay_parameters
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in named_parameters
                            if n in decay_parameters and not self._is_mot_action_branch_param(n)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in named_parameters
                            if n in decay_parameters and self._is_mot_action_branch_param(n)
                        ],
                        "weight_decay": action_expert_weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in named_parameters
                            if n not in decay_parameters
                        ],
                        "weight_decay": 0.0,
                    },
                ]
                if self.global_rank == 0:
                    print(
                        "Using action_expert_weight_decay="
                        f"{action_expert_weight_decay:g}; other decay parameters keep "
                        f"weight_decay={self.args.weight_decay:g}"
                    )
            optimizer_grouped_parameters = [
                group for group in optimizer_grouped_parameters if group["params"]
            ]

            optimizer_cls, optimizer_kwargs = transformers.Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            # DeepSpeed CPU Adam (ZeRO offload) expects 'bias_correction' in each param group.
            # HuggingFace Trainer's AdamW does not set it, causing KeyError in cpu_adam.step().
            if getattr(self.args, "deepspeed", None):
                for group in self.optimizer.param_groups:
                    group.setdefault("bias_correction", True)

        return self.optimizer

    def save_model(self, output_dir: Optional[str], _internal_call: bool):
        # Only the saving rank should materialize a full state_dict. Under ZeRO-2,
        # doing this on every rank creates a large transient CPU-memory spike and can
        # cause the OS to kill dataloader workers during checkpoint save.
        if not self.args.should_save:
            return

        ## save tuned model separately
        if self.is_deepspeed_enabled:
            state_dict = self.accelerator.get_state_dict(self.deepspeed)
        else:
            state_dict = self.model.state_dict()

        if self.base_cfg.save_lora_only:
            # Save only the trainable parameters
            train_key = [k for k, v in self.model.named_parameters() if v.requires_grad]
            lora_state_dict = {k: v for k, v in self.model.state_dict().items() if k in train_key}
            state_dict = lora_state_dict

        ret = self.model.save_pretrained(output_dir, state_dict=state_dict)
        sidecars = materialize_component_sidecars(self.model, output_dir)
        if sidecars:
            mprint(f"Materialized local component sidecars into {output_dir}: {sidecars}")

        # can separately save the VLM model for downstream evalualtion
        if self.base_cfg.save_llm:
            llm_output_dir = os.path.join(output_dir, "llm")
            self.model.backbone.model.save_pretrained(llm_output_dir)

        if self.base_cfg.save_value_model:
            assert hasattr(
                self.model.action_head, "value_model"
            ), f"Value model not found in action head: {type(self.model.action_head)}"
            value_model_output_dir = os.path.join(output_dir, "value_model")
            self.model.action_head.value_model.save_pretrained(value_model_output_dir)

        # Free the temporary consolidated weights immediately after checkpoint save.
        del state_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return ret

    def train(
        self,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
        **kwargs,
    ):
        """Correctly set self.state from checkpoint so get_train_dataloader can read from it."""
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(
                    f"No valid checkpoint found in output directory ({self.args.output_dir})"
                )

        if resume_from_checkpoint is not None:
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )
        return super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if not isinstance(train_dataset, (ShardedLeRobotMixtureDataset)):
            return super().get_train_dataloader()

        # During resume, don't skip the data
        self.args.ignore_data_skip = True
        curr_global_step = self.state.global_step
        print(f"Current global step: {curr_global_step}")
        if curr_global_step > 0:
            new_seed = train_dataset.seed + curr_global_step
            train_dataset.reset_seed(new_seed)
            print(
                f"Resetting seed to {new_seed}. Please note that this will make the experiment non-reproducible."
            )

        print("Creating custom train dataloader")
        # Handle the case where the dataset is an IterableDataset
        data_collator = self.data_collator
        data_collator = self._get_collator_with_removed_columns(
            data_collator, description="training"
        )

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "drop_last": getattr(self.args, "dataloader_drop_last", False),
        }
        # These arguments are only valid when num_workers > 0 (PyTorch raises otherwise).
        if self.args.dataloader_num_workers > 0:
            dataloader_params["persistent_workers"] = self.args.dataloader_persistent_workers
            dataloader_prefetch_factor = getattr(
                self.args, "dataloader_prefetch_factor", None
            )
            if dataloader_prefetch_factor is not None:
                dataloader_params["prefetch_factor"] = dataloader_prefetch_factor
            if "in_order" in inspect.signature(DataLoader).parameters:
                dataloader_params["in_order"] = self.dataloader_in_order

        return DataLoader(train_dataset, **dataloader_params)


class BaseExperiment(ABC):
    def __init__(self, cfg: DictConfig):
        # assert cfg.save_steps == 500, "save_steps must be 500 for standarized evaluation"
        assert cfg.max_steps > 0, "max_steps must be > 0 for standarized evaluation"
        assert cfg.save_total_limit >= 5, "save_total_limit must be >= 5 for standarized evaluation"

        if cfg.load_from_yaml is not None:
            # Override the default config with the loaded config.
            loaded_cfg = OmegaConf.load(cfg.load_from_yaml)
            cfg = loaded_cfg  # overwrite

        # Check if evaluation transforms are valid.
        assert cfg.transforms is not None, "Evaluation transforms are not provided."
        for tag, transform_cfg in cfg.transforms.items():
            try:
                # Check if the tag is a valid EmbodimentTag
                _ = EmbodimentTag(tag)
                # Check if the transform is a valid ComposedModalityTransform
                transform = instantiate(transform_cfg)
                assert isinstance(transform, ComposedModalityTransform), f"{transform=}"
            except Exception as e:
                raise ValueError(f"Evaluation transform {tag} is invalid: {e}")

        # Instantiate the training arguments.
        cfg.training_args.output_dir = cfg.training_args.output_dir.rstrip("/")
        cfg.training_args.run_name = cfg.training_args.output_dir.split("/")[-1]
        print(f"Run name: {cfg.training_args.run_name}")
        training_args = instantiate(cfg.training_args)
        set_seed(training_args.seed)

        # Optionally let SwanLab mirror wandb logs before wandb is initialized by HF Trainer.
        maybe_enable_swanlab_sync()

        # Set the environment variables for wandb.
        if "WANDB_PROJECT" not in os.environ:
            os.environ["WANDB_PROJECT"] = cfg.wandb_project
        if "WANDB_RUN_ID" not in os.environ:
            runtime_id = os.environ.get("RUNTIME_ID", None)
            """If a RUNTIME_ID is available in the environment, we use it as the wandb id,
            which will allow to display the evaluation results and the training results
            in the same wandb run. Otherwise, we create a new run."""
            if runtime_id:
                os.environ["WANDB_RUN_ID"] = runtime_id
        os.environ["WANDB_DIR"] = training_args.output_dir

        # Create the experiment config directory.
        output_dir = Path(training_args.output_dir)
        exp_cfg_dir = output_dir / "experiment_cfg"
        exp_cfg_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, exp_cfg_dir / "conf.yaml", resolve=True)

        wandb_config_file = output_dir / "wandb_config.json"
        with open(wandb_config_file, "w") as f:
            json.dump(
                {
                    "project": os.environ.get("WANDB_PROJECT", ""),
                    "run_id": os.environ.get("WANDB_RUN_ID", ""),
                },
                f,
            )

        # Check if we are resuming training.
        resume_path, continue_training = get_checkpoint_path(training_args.output_dir)
        if not continue_training:
            print(f"Models is ready under {training_args.output_dir}. Skip training.")
            exit(0)
        if resume_path:
            print(f"Resuming training from {resume_path}")
            resume_from_checkpoint = True
        else:
            # First time training.
            resume_from_checkpoint = False

        # Instantiate the model.
        model = self.create_model(cfg, training_args)

        if hasattr(model.action_head, "max_steps"):
            model.action_head.max_steps = cfg.max_steps

        # Make sure model_dtype and training_args dtype are compatible.
        compute_dtype = dtype_from_string(model.config.model_dtype)

        # Create the train dataset.
        # Dump the metadata; necessary for policy to normalize the input and unnormalize the output
        train_dataset = self.create_train_dataset(cfg, model)
        print("Using dataset:")
        print(train_dataset)
        assert (
            train_dataset.merged_metadata is not None
        ), "You must set metadata_config.merge=true in order to save the metadata."

        metadata_save_path = exp_cfg_dir / "metadata.json"
        U.json_dump(
            {k: v.model_dump(mode="json") for k, v in train_dataset.merged_metadata.items()},
            metadata_save_path,
            indent=4,
        )
        print("Successfully dumped metadata")

        val_dataset = self.create_val_dataset(cfg, model)
        data_collator = self.create_data_collator(cfg, model)
        trainer = self.create_trainer(
            cfg=cfg,
            exp_cfg_dir=exp_cfg_dir,
            model=model,
            training_args=training_args,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            data_collator=data_collator,
            compute_dtype=compute_dtype,
        )
        self.cfg = cfg
        self.exp_cfg_dir = exp_cfg_dir
        self.training_args = training_args
        self.resume_from_checkpoint = resume_from_checkpoint
        self.train_dataset = train_dataset
        self.trainer = trainer

    def create_model(self, cfg, training_args):
        if cfg.pretrained_model_path is not None:
            prepare_action_head_cfg_for_checkpoint(
                cfg.model.config.action_head_cfg,
                cfg.pretrained_model_path,
            )
        model = instantiate(cfg.model)

        if cfg.pretrained_model_path is not None:
            mprint(f"Loading pretrained weights from: {cfg.pretrained_model_path}")
            import json, gc
            from safetensors.torch import load_file

            ckpt_dir = cfg.pretrained_model_path
            safetensors_index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
            safetensors_path = os.path.join(ckpt_dir, "model.safetensors")

            if os.path.exists(safetensors_index_path):
                with open(safetensors_index_path, 'r') as f:
                    index = json.load(f)
                for shard_file in sorted(set(index["weight_map"].values())):
                    shard_path = os.path.join(ckpt_dir, shard_file)
                    mprint(f"Loading shard: {shard_path}")
                    shard_state_dict = load_file(shard_path)
                    model.load_state_dict(shard_state_dict, strict=False)
                    del shard_state_dict
                    gc.collect()
            elif os.path.exists(safetensors_path):
                state_dict = load_file(safetensors_path)
                model.load_state_dict(state_dict, strict=False)
            else:
                raise FileNotFoundError(
                    f"No weights found at '{ckpt_dir}'. "
                    "Expected 'model.safetensors' or 'model.safetensors.index.json'."
                )

            if (hasattr(model, 'action_head')
                    and hasattr(model.action_head, 'inject_lora_after_loading')
                    and model.action_head.config.defer_lora_injection):
                model.action_head.inject_lora_after_loading()

            mprint("Successfully loaded pretrained weights")

        model.config.resume_path = model.config._name_or_path = training_args.output_dir
        # Avoid printing the full model structure during training startup.
        return model

    def create_train_dataset(self, cfg, model):
        assert torch.distributed.is_initialized()
        train_dataset = instantiate(cfg.train_dataset)
        return train_dataset

    def create_val_dataset(self, cfg, model):
        if "val_dataset" in cfg and cfg.val_dataset is not None:
            return instantiate(cfg.val_dataset)
        return None

    def create_data_collator(self, cfg, model):
        return instantiate(cfg.data_collator)

    def create_trainer(
        self,
        cfg,
        exp_cfg_dir,
        model,
        training_args,
        train_dataset,
        val_dataset,
        data_collator,
        compute_dtype,
    ):
        # Set the gradient accumulation steps.
        if cfg.global_batch_size is not None:
            global_bs = cfg.global_batch_size
            bs = training_args.per_device_train_batch_size
            grad_acc = compute_grad_accum_to_match_global_bs(global_bs, bs)
            training_args.gradient_accumulation_steps = grad_acc
            print(
                f"Set global batch size to {global_bs}, set gradient accumulation steps to {grad_acc}"
            )
        elif cfg.raise_error_if_global_batch_size_not_set:
            raise ValueError(
                "global_batch_size is not set. To ensure the scripts can be reproduced regardless of the number of nodes used, please set this."
            )
        else:
            warnings.warn(
                "global_batch_size is not set. This is fine for debugging, but please set this for real experiments."
            )

        # Instantiate the partial trainer.
        trainer_partial = instantiate(
            cfg.trainer,
            model=model,
            output_dir=training_args.output_dir,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_dtype=compute_dtype,
        )

        # Fully instantiate the trainer with dataclasses instances.
        trainer = trainer_partial(data_collator=data_collator, args=training_args)
        trainer.base_cfg = cfg
        train_dl_len = len(trainer.get_train_dataloader())
        eval_dl_len = (
            len(trainer.get_eval_dataloader()) if val_dataset is not None else "no eval dataloader"
        )

        # Save the total training steps in the config.
        with open_dict(cfg):
            cfg.total_training_steps = train_dl_len * cfg.training_args.num_train_epochs

        # Save config.
        OmegaConf.save(cfg, exp_cfg_dir / "conf.yaml", resolve=True)

        run_name = cfg.training_args.get("run_name", None)
        ckpt_format_callback = CheckpointFormatCallback(run_name=run_name, exp_cfg_dir=exp_cfg_dir)
        trainer.add_callback(ckpt_format_callback)

        loss_log_path = str(Path(training_args.output_dir) / "loss_log.jsonl")
        trainer.add_callback(LossLoggerCallback(output_path=loss_log_path))
        trainer.add_callback(NVTXTrainerCallback())


        # Add profiling callback (local profiling only, no S3 upload)
        # Local: {output_dir}/profiling/rank_{id}/*.pt.trace.json
        if cfg.trainer.get("enable_prof_callback", False):
            output_dir = Path(training_args.output_dir)
            global_rank = int(os.environ.get("RANK", "0"))

            # Get profiling configuration from trainer config
            profile_start_step = cfg.trainer.get("profile_start_step", 50)
            profile_warmup_steps = cfg.trainer.get("profile_warmup_steps", 1)
            profile_active_steps = cfg.trainer.get("profile_active_steps", 5)
            profile_record_shapes = cfg.trainer.get("profile_record_shapes", False)
            profile_with_stack = cfg.trainer.get(
                "profile_with_stack", False
            )  # Default False to match omni (stack traces add significant file size)
            profile_memory = cfg.trainer.get("profile_memory", False)

            # Calculate end step
            profile_end_step = profile_start_step + profile_warmup_steps + profile_active_steps - 1

            # Setup profile directory with rank subdirectory: {output_dir}/profiling/rank_{id}/
            profile_dir = output_dir / "profiling" / f"rank_{global_rank}"
            profile_dir.mkdir(parents=True, exist_ok=True)

            mprint(
                f"Profiling enabled: steps {profile_start_step}-{profile_end_step}, "
                f"saving to {profile_dir}"
            )

            # Add ProfCallback
            trainer.add_callback(
                ProfCallback(
                    profile_dir=profile_dir,
                    upload_callback=None,
                    profile_start_step=profile_start_step,
                    profile_end_step=profile_end_step,
                    warmup_steps=profile_warmup_steps,
                    active_steps=profile_active_steps,
                    trainer=trainer,
                    record_shapes=profile_record_shapes,
                    with_stack=profile_with_stack,
                    profile_memory=profile_memory,
                )
            )

        mprint(
            f"train dataloader length: {train_dl_len}\n"
            f"eval dataloader length: {eval_dl_len}\n"
            f"train dataset length: {len(trainer.train_dataset)}\n"
            f"GPU memory before training: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024} GB",
            flush=True,
        )
        return trainer

    def train(self):
        # Start training.
        self.trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)
        if os.environ.get("DREAMZERO_SKIP_FINAL_SAVE", "").strip().lower() in {"1", "true", "yes", "on"}:
            mprint("DREAMZERO_SKIP_FINAL_SAVE is enabled; skipping final Trainer state/model save.")
            return
        self.trainer.save_state()
        safe_save_model_for_hf_trainer(
            trainer=self.trainer, output_dir=self.training_args.output_dir
        )
