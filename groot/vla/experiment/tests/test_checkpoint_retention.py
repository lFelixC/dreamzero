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

"""Test checkpoint retention logic."""

import os
import shutil
from pathlib import Path
from typing import Optional
import unittest

from omegaconf import OmegaConf


class CheckpointRetentionTest(unittest.TestCase):
    """Test checkpoint retention logic in BaseTrainer."""

    def setUp(self):
        """Create a temporary directory for test checkpoints."""
        self.test_dir = Path("/tmp/test_checkpoint_retention")
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

    def test_protected_checkpoints_basic(self):
        """Test that protected checkpoints are not deleted."""
        # Import after setting up test directory
        from groot.vla.experiment.checkpoint_retention import CheckpointRetention

        # Create checkpoints
        for step in [1000, 2000, 3000, 4000, 5000, 6000, 7000]:
            self._create_checkpoint(step)

        # Configure retention
        config = OmegaConf.create({
            "save_total_limit": 3,  # Only keep 3 recent
            "checkpoint_retention": {
                "keep_every_n_steps": 5000,  # But keep every 5000
                "keep_milestone_steps": [1000],  # And keep step 1000
            }
        })

        retention = CheckpointRetention(config, self.test_dir)

        # Get protected steps
        protected = retention.get_protected_steps(7000)

        # Should include: 1000 (milestone), 5000 (every_n)
        self.assertIn(1000, protected)
        self.assertIn(5000, protected)
        self.assertEqual(len(protected), 2)

    def test_keep_every_n_steps(self):
        """Test keep_every_n_steps retention rule."""
        from groot.vla.experiment.checkpoint_retention import CheckpointRetention

        config = OmegaConf.create({
            "save_total_limit": 3,
            "checkpoint_retention": {
                "keep_every_n_steps": 20000,
            }
        })

        retention = CheckpointRetention(config, self.test_dir)

        # At step 50000, should protect: 20000, 40000
        protected = retention.get_protected_steps(50000)
        self.assertIn(20000, protected)
        self.assertIn(40000, protected)
        self.assertEqual(len(protected), 2)

        # At step 60000, should protect: 20000, 40000, 60000
        protected = retention.get_protected_steps(60000)
        self.assertIn(20000, protected)
        self.assertIn(40000, protected)
        self.assertIn(60000, protected)
        self.assertEqual(len(protected), 3)

    def test_keep_milestone_steps(self):
        """Test keep_milestone_steps retention rule."""
        from groot.vla.experiment.checkpoint_retention import CheckpointRetention

        config = OmegaConf.create({
            "save_total_limit": 3,
            "checkpoint_retention": {
                "keep_milestone_steps": [5000, 10000, 50000, 100000],
            }
        })

        retention = CheckpointRetention(config, self.test_dir)

        # At step 7000, should protect: 5000
        protected = retention.get_protected_steps(7000)
        self.assertIn(5000, protected)
        self.assertNotIn(10000, protected)
        self.assertEqual(len(protected), 1)

        # At step 10000, should protect: 5000, 10000
        protected = retention.get_protected_steps(10000)
        self.assertIn(5000, protected)
        self.assertIn(10000, protected)
        self.assertEqual(len(protected), 2)

    def test_combined_retention_rules(self):
        """Test that multiple retention rules work together."""
        from groot.vla.experiment.checkpoint_retention import CheckpointRetention

        config = OmegaConf.create({
            "save_total_limit": 3,
            "checkpoint_retention": {
                "keep_every_n_steps": 20000,
                "keep_milestone_steps": [5000, 10000],
            }
        })

        retention = CheckpointRetention(config, self.test_dir)

        # At step 30000, should protect:
        # - every_n: 20000
        # - milestones: 5000, 10000
        protected = retention.get_protected_steps(30000)
        self.assertIn(20000, protected)
        self.assertIn(5000, protected)
        self.assertIn(10000, protected)
        self.assertEqual(len(protected), 3)

    def test_empty_retention_config(self):
        """Test that empty checkpoint_retention config works."""
        from groot.vla.experiment.checkpoint_retention import CheckpointRetention

        config = OmegaConf.create({
            "save_total_limit": 3,
            "checkpoint_retention": {},
        })

        retention = CheckpointRetention(config, self.test_dir)
        protected = retention.get_protected_steps(10000)

        # No protected steps
        self.assertEqual(len(protected), 0)

    def test_no_retention_config(self):
        """Test that missing checkpoint_retention config works."""
        from groot.vla.experiment.checkpoint_retention import CheckpointRetention

        config = OmegaConf.create({
            "save_total_limit": 3,
        })

        retention = CheckpointRetention(config, self.test_dir)
        protected = retention.get_protected_steps(10000)

        # No protected steps
        self.assertEqual(len(protected), 0)

    def test_null_retention_config(self):
        """Test that null checkpoint_retention config works."""
        from groot.vla.experiment.checkpoint_retention import CheckpointRetention

        config = OmegaConf.create({
            "save_total_limit": 3,
            "checkpoint_retention": None,
        })

        retention = CheckpointRetention(config, self.test_dir)
        protected = retention.get_protected_steps(10000)

        # No protected steps when config is None
        self.assertEqual(len(protected), 0)

    def test_is_protected_checkpoint(self):
        """Test is_protected_checkpoint method."""
        from groot.vla.experiment.checkpoint_retention import CheckpointRetention

        config = OmegaConf.create({
            "save_total_limit": 3,
            "checkpoint_retention": {
                "keep_every_n_steps": 10000,
                "keep_milestone_steps": [5000],
            }
        })

        retention = CheckpointRetention(config, self.test_dir)

        # Create some checkpoint directories
        for step in [1000, 5000, 10000, 15000]:
            self._create_checkpoint(step)

        # Test protected checkpoints
        self.assertTrue(retention.is_protected_checkpoint(str(self.test_dir / "checkpoint-5000"), 20000))
        self.assertTrue(retention.is_protected_checkpoint(str(self.test_dir / "checkpoint-10000"), 20000))

        # Test non-protected checkpoints
        self.assertFalse(retention.is_protected_checkpoint(str(self.test_dir / "checkpoint-1000"), 20000))
        self.assertFalse(retention.is_protected_checkpoint(str(self.test_dir / "checkpoint-15000"), 20000))

    def test_get_checkpoints_to_delete(self):
        """Test that get_checkpoints_to_delete skips protected checkpoints."""
        from groot.vla.experiment.checkpoint_retention import CheckpointRetention

        config = OmegaConf.create({
            "save_total_limit": 3,
            "checkpoint_retention": {
                "keep_every_n_steps": 20000,
                "keep_milestone_steps": [10000],
            }
        })

        retention = CheckpointRetention(config, self.test_dir)

        # Create many checkpoints
        steps = [1000, 5000, 10000, 15000, 20000, 25000, 30000, 35000]
        for step in steps:
            self._create_checkpoint(step)

        current_step = 35000
        all_checkpoints = [str(self.test_dir / f"checkpoint-{step}") for step in steps]

        # Get checkpoints to delete with limit of 3
        to_delete = retention.get_checkpoints_to_delete(
            all_checkpoints, current_step, limit=3
        )

        # Protected: 10000 (milestone), 20000 (every_n)
        # Non-protected: 1000, 5000, 15000, 25000, 35000
        # Should delete the 3 oldest non-protected: 1000, 5000, 15000
        self.assertIn(str(self.test_dir / "checkpoint-1000"), to_delete)
        self.assertIn(str(self.test_dir / "checkpoint-5000"), to_delete)
        self.assertIn(str(self.test_dir / "checkpoint-15000"), to_delete)
        self.assertEqual(len(to_delete), 3)

        # Protected checkpoints should NOT be in delete list
        self.assertNotIn(str(self.test_dir / "checkpoint-10000"), to_delete)
        self.assertNotIn(str(self.test_dir / "checkpoint-20000"), to_delete)

    def test_keep_best_n_placeholder(self):
        """Test placeholder for keep_best_n (will be implemented with eval metrics)."""
        from groot.vla.experiment.checkpoint_retention import CheckpointRetention

        config = OmegaConf.create({
            "save_total_limit": 3,
            "checkpoint_retention": {
                "keep_best_n": 0,  # Disabled for now
            }
        })

        retention = CheckpointRetention(config, self.test_dir)
        protected = retention.get_protected_steps(10000)

        # No protected steps since keep_best_n is disabled/0
        self.assertEqual(len(protected), 0)


if __name__ == "__main__":
    unittest.main()
