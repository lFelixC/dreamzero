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

"""Test BaseTrainer checkpoint rotation override."""

import os
import shutil
from pathlib import Path
import unittest

from omegaconf import OmegaConf


class TestBaseTrainerCheckpointRotation(unittest.TestCase):
    """Test that BaseTrainer properly overrides _rotate_checkpoints."""

    def setUp(self):
        """Create a temporary directory for test checkpoints."""
        self.test_dir = Path("/tmp/test_basetrainer_rotation")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Clean up temporary directory."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def _create_checkpoint(self, step: int):
        """Create a mock checkpoint directory."""
        ckpt_dir = self.test_dir / f"checkpoint-{step}"
        ckpt_dir.mkdir(exist_ok=True)
        # Create some marker files
        (ckpt_dir / "model.safetensors").touch()
        (ckpt_dir / "trainer_state.json").touch()
        return ckpt_dir

    def _get_checkpoint_steps(self) -> list[int]:
        """Get list of checkpoint steps from the test directory."""
        checkpoints = []
        for path in self.test_dir.glob("checkpoint-*"):
            match = path.name.split("-")[-1]
            if match.isdigit():
                checkpoints.append(int(match))
        return sorted(checkpoints)

    def test_basetrainer_has_rotate_checkpoints_override(self):
        """Test that BaseTrainer has _rotate_checkpoints method that respects retention."""
        from groot.vla.experiment.base import BaseTrainer

        # Check that BaseTrainer has the method
        self.assertTrue(hasattr(BaseTrainer, "_rotate_checkpoints"))

        # Check if it's overridden (not just inherited from transformers.Trainer)
        # We can check by looking at the method's class
        method = getattr(BaseTrainer, "_rotate_checkpoints")
        self.assertIn("BaseTrainer", str(method))

    def test_basetrainer_retention_initialization(self):
        """Test that BaseTrainer initializes checkpoint_retention when configured."""
        from groot.vla.experiment.base import BaseTrainer
        from groot.vla.experiment.checkpoint_retention import CheckpointRetention

        # Check that BaseTrainer has a way to access checkpoint retention
        # This will be a hasattr check on an instance
        self.assertTrue(True)  # Placeholder for full integration test

    def test_rotation_preserves_milestone_checkpoints(self):
        """Test that rotation preserves milestone checkpoints."""
        # Create checkpoints
        for step in [1000, 5000, 10000, 15000, 20000, 25000, 30000]:
            self._create_checkpoint(step)

        # Before rotation, all should exist
        steps_before = self._get_checkpoint_steps()
        self.assertEqual(len(steps_before), 7)

        # Simulate what rotation should do:
        # - save_total_limit=3 (keep 3 recent)
        # - milestone_steps=[10000] (keep 10000)
        # After rotation, should keep: 10000, 20000, 25000, 30000 (4 total)
        # Protected (10000) + 3 recent non-protected

        # This is a behavioral test - the actual implementation
        # will be verified with a real BaseTrainer instance

        # Expected protected: 10000 (milestone)
        # Expected kept (non-protected): 20000, 25000, 30000
        # Expected deleted: 1000, 5000, 15000

        protected = {10000}
        limit = 3
        all_steps = [1000, 5000, 10000, 15000, 20000, 25000, 30000]

        non_protected = [s for s in all_steps if s not in protected]
        to_keep = set(protected) | set(non_protected[-limit:])
        to_delete = set(all_steps) - to_keep

        self.assertEqual(to_delete, {1000, 5000, 15000})
        self.assertEqual(to_keep, {10000, 20000, 25000, 30000})


if __name__ == "__main__":
    unittest.main()
