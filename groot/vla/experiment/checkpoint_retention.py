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

"""Checkpoint retention logic for long-term checkpoint management.

This module implements a two-tier checkpoint retention system:
1. High-frequency temporary checkpoints (save_steps + save_total_limit)
2. Long-term retention with protection rules (keep_every_n_steps, keep_milestone_steps, etc.)
"""

import logging
import re
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


class CheckpointRetention:
    """Manages long-term checkpoint retention rules.

    Protected checkpoints are excluded from the standard save_total_limit rotation,
    allowing them to be kept indefinitely while still respecting the recent N limit
    for temporary checkpoints.
    """

    CHECKPOINT_DIR_PREFIX = "checkpoint"

    def __init__(self, config: DictConfig, output_dir: Path):
        """Initialize checkpoint retention manager.

        Args:
            config: Training configuration with checkpoint_retention settings
            output_dir: Directory where checkpoints are saved
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.retention_config = self._get_retention_config(config)

    def _get_retention_config(self, config: DictConfig) -> dict:
        """Extract checkpoint_retention config from main config.

        Returns empty dict if checkpoint_retention is not configured or is None.
        """
        if isinstance(config, dict):
            cfg = config.get("checkpoint_retention", {})
            return cfg if cfg is not None else {}
        elif hasattr(config, "checkpoint_retention"):
            if config.checkpoint_retention is None:
                return {}
            return OmegaConf.to_container(config.checkpoint_retention, resolve=True)
        return {}

    def get_protected_steps(self, current_step: int) -> set[int]:
        """Get set of checkpoint steps that should be protected from deletion.

        A checkpoint is protected if it matches any of the configured retention rules:
        - keep_every_n_steps: Keep checkpoints at multiples of this interval
        - keep_milestone_steps: Keep checkpoints at explicitly specified steps
        - keep_best_n: (Future) Keep N best checkpoints based on eval metrics

        Args:
            current_step: Current training step

        Returns:
            Set of step numbers that should be protected from deletion
        """
        protected = set()

        # keep_every_n_steps: protect checkpoints at regular intervals
        keep_every_n = self.retention_config.get("keep_every_n_steps")
        if keep_every_n and keep_every_n > 0:
            # Protect all multiples of keep_every_n up to current_step
            for step in range(keep_every_n, current_step + 1, keep_every_n):
                protected.add(step)

        # keep_milestone_steps: protect explicitly specified milestones
        milestone_steps = self.retention_config.get("keep_milestone_steps", [])
        if milestone_steps:
            if isinstance(milestone_steps, int):
                milestone_steps = [milestone_steps]
            for step in milestone_steps:
                if step <= current_step:
                    protected.add(step)

        # keep_best_n: (Future) would protect best N checkpoints based on eval metrics
        # This requires tracking eval metrics, which will be added later

        return protected

    def is_protected_checkpoint(self, checkpoint_path: str, current_step: int) -> bool:
        """Check if a checkpoint is protected from deletion.

        Args:
            checkpoint_path: Path to the checkpoint directory
            current_step: Current training step

        Returns:
            True if checkpoint should be protected from deletion
        """
        step = self._extract_step_from_path(checkpoint_path)
        if step is None:
            return False

        protected_steps = self.get_protected_steps(current_step)
        return step in protected_steps

    def _extract_step_from_path(self, checkpoint_path: str) -> Optional[int]:
        """Extract step number from checkpoint directory path.

        Args:
            checkpoint_path: Path to checkpoint directory (e.g., "/path/to/checkpoint-5000")

        Returns:
            Step number as int, or None if not a valid checkpoint path
        """
        # Match pattern: checkpoint-{number}
        match = re.search(rf"{self.CHECKPOINT_DIR_PREFIX}-(\d+)", checkpoint_path)
        if match:
            return int(match.group(1))
        return None

    def get_checkpoints_to_delete(
        self,
        all_checkpoints: list[str],
        current_step: int,
        limit: Optional[int] = None,
    ) -> list[str]:
        """Get list of checkpoints that can be deleted, respecting protected checkpoints.

        Args:
            all_checkpoints: List of all checkpoint directory paths, sorted oldest to newest
            current_step: Current training step
            limit: Maximum number of checkpoints to keep (from non-protected checkpoints)

        Returns:
            List of checkpoint paths that should be deleted
        """
        if limit is None or limit <= 0:
            return []

        protected_steps = self.get_protected_steps(current_step)

        # Separate checkpoints into protected and non-protected
        protected_checkpoints = []
        non_protected_checkpoints = []

        for ckpt in all_checkpoints:
            step = self._extract_step_from_path(ckpt)
            if step is not None and step in protected_steps:
                protected_checkpoints.append(ckpt)
            else:
                non_protected_checkpoints.append(ckpt)

        # Keep only the N most recent non-protected checkpoints
        # (all_checkpoints is already sorted oldest to newest)
        # Get the most recent checkpoints (including protected)
        most_recent_checkpoints = all_checkpoints[-limit:] if limit < len(all_checkpoints) else all_checkpoints[:]

        # Check which of the recent checkpoints are protected
        recent_protected = []
        recent_non_protected = []
        for ckpt in most_recent_checkpoints:
            step = self._extract_step_from_path(ckpt)
            if step is not None and step in protected_steps:
                recent_protected.append(ckpt)
            else:
                recent_non_protected.append(ckpt)

        # All other checkpoints (not in the most recent 'limit') can be deleted,
        # EXCEPT for protected checkpoints which must always be kept
        to_delete = []
        for ckpt in all_checkpoints[:-limit] if limit < len(all_checkpoints) else []:
            step = self._extract_step_from_path(ckpt)
            if step is not None and step in protected_steps:
                # Protected checkpoint, don't delete
                continue
            to_delete.append(ckpt)

        logger.info(
            f"Checkpoint retention: keeping {len(protected_checkpoints)} protected + "
            f"{limit} recent non-protected checkpoints. Deleting {len(to_delete)} old checkpoints."
        )

        return to_delete
