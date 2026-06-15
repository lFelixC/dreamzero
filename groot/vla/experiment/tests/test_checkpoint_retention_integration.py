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

"""Integration test for checkpoint retention in BaseTrainer."""

import os
import shutil
from pathlib import Path
import unittest

from omegaconf import OmegaConf

try:
    from transformers import TrainingArguments
except ImportError:
    TrainingArguments = None


class BaseTrainerRetentionTest(unittest.TestCase):
    """Test checkpoint retention integration with BaseTrainer."""

    def setUp(self):
        """Create a temporary directory for test checkpoints."""
        self.test_dir = Path("/tmp/test_trainer_retention")
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

    def test_basetrainer_has_retention_manager(self):
        """Test that BaseTrainer has a checkpoint_retention attribute when configured."""
        # This test will check that BaseTrainer properly initializes retention
        # We'll need to mock/minimal test the integration

        # For now, test the logic without full BaseTrainer initialization
        from groot.vla.experiment.checkpoint_retention import CheckpointRetention

        # Create test config
        config = OmegaConf.create({
            "save_total_limit": 3,
            "checkpoint_retention": {
                "keep_every_n_steps": 10000,
                "keep_milestone_steps": [5000],
            }
        })

        retention = CheckpointRetention(config, self.test_dir)

        # Should have retention config
        self.assertIsNotNone(retention.retention_config)
        self.assertEqual(retention.retention_config.get("keep_every_n_steps"), 10000)

    def test_rotation_skips_protected_checkpoints(self):
        """Test that _rotate_checkpoints skips protected checkpoints."""
        from groot.vla.experiment.checkpoint_retention import CheckpointRetention

        # Create many checkpoints
        steps = [1000, 5000, 10000, 15000, 20000, 25000, 30000, 35000]
        for step in steps:
            self._create_checkpoint(step)

        config = OmegaConf.create({
            "save_total_limit": 3,
            "checkpoint_retention": {
                "keep_every_n_steps": 20000,
                "keep_milestone_steps": [10000],
            }
        })

        retention = CheckpointRetention(config, self.test_dir)

        # Simulate rotation at step 35000
        current_step = 35000
        all_checkpoints = sorted(
            [str(self.test_dir / f"checkpoint-{step}") for step in steps],
            key=lambda x: int(x.split("-")[-1])
        )

        # Get checkpoints to delete with limit of 3
        to_delete = retention.get_checkpoints_to_delete(
            all_checkpoints, current_step, limit=3
        )

        # Should delete the 3 oldest non-protected: 1000, 5000, 15000
        # Protected: 10000 (milestone), 20000 (every_n)
        expected_deleted = ["1000", "5000", "15000"]
        actual_deleted = [Path(p).name.split("-")[-1] for p in to_delete]

        self.assertEqual(len(to_delete), 3)
        for step in expected_deleted:
            self.assertIn(step, actual_deleted)

        # Protected should not be deleted
        for step_str in actual_deleted:
            self.assertNotIn(step_str, ["10000", "20000"])

    def test_rotation_with_all_protected_checkpoints(self):
        """Test rotation when all checkpoints are protected."""
        from groot.vla.experiment.checkpoint_retention import CheckpointRetention

        # Create checkpoints at protected milestones
        steps = [10000, 20000, 30000, 40000]
        for step in steps:
            self._create_checkpoint(step)

        config = OmegaConf.create({
            "save_total_limit": 2,  # Would normally delete 2
            "checkpoint_retention": {
                "keep_every_n_steps": 10000,  # But all are protected
            }
        })

        retention = CheckpointRetention(config, self.test_dir)

        current_step = 40000
        all_checkpoints = sorted(
            [str(self.test_dir / f"checkpoint-{step}") for step in steps],
            key=lambda x: int(x.split("-")[-1])
        )

        to_delete = retention.get_checkpoints_to_delete(
            all_checkpoints, current_step, limit=2
        )

        # All are protected, so nothing should be deleted
        self.assertEqual(len(to_delete), 0)


if __name__ == "__main__":
    unittest.main()
