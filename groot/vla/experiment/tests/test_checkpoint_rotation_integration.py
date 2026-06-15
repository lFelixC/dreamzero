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

"""Integration test for BaseTrainer._rotate_checkpoints with actual directory operations."""

import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import Mock

from omegaconf import OmegaConf
from transformers import TrainingArguments

from groot.vla.experiment.base import BaseTrainer


class MinimalTrainer(BaseTrainer):
    """Minimal trainer subclass for testing checkpoint rotation."""

    def __init__(self, *args, **kwargs):
        # Set minimal required attributes
        self.base_cfg = kwargs.pop("base_cfg", None)
        super().__init__(*args, **kwargs)


class TestCheckpointRotationIntegration(unittest.TestCase):
    """Test _rotate_checkpoints with actual directory deletion."""

    def setUp(self):
        """Create a temporary directory for test checkpoints."""
        self.test_dir = Path("/tmp/test_rotation_integration")
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
        (ckpt_dir / "config.json").touch()
        return ckpt_dir

    def _get_checkpoint_steps(self) -> list[int]:
        """Get list of checkpoint steps from the test directory."""
        checkpoints = []
        for path in self.test_dir.glob("checkpoint-*"):
            match = path.name.split("-")[-1]
            if match.isdigit():
                checkpoints.append(int(match))
        return sorted(checkpoints)

    def _create_minimal_trainer(self, base_cfg):
        """Create a minimal trainer for testing."""
        # Create a tiny dummy model and dataset
        import torch
        import torch.nn as nn

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 2)

            def forward(self, x):
                return {"loss": torch.tensor(0.5)}

        class DummyDataset:
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return {"input_ids": torch.randn(10)}

        model = DummyModel()
        train_dataset = DummyDataset()

        args = TrainingArguments(
            output_dir=str(self.test_dir),
            save_total_limit=3,
            save_strategy="steps",
            save_steps=1000,
            logging_steps=100,
            max_steps=1000,
            report_to="none",
        )

        # Mock for compute_dtype (required by BaseTrainer)
        import torch
        compute_dtype = torch.float32

        trainer = MinimalTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            compute_dtype=compute_dtype,
            output_dir=str(self.test_dir),
            base_cfg=base_cfg,
        )

        # Mock the state
        trainer.state = Mock()
        trainer.state.global_step = 35000
        trainer.state.best_model_checkpoint = None

        return trainer

    def test_rotation_deletes_actual_directories(self):
        """Test that _rotate_checkpoints actually deletes directories."""
        from groot.vla.experiment.checkpoint_retention import CheckpointRetention

        # Create checkpoints
        steps = [1000, 2000, 3000, 4000, 5000, 6000]
        for step in steps:
            self._create_checkpoint(step)

        # Verify all exist
        self.assertEqual(self._get_checkpoint_steps(), steps)

        # Create trainer with retention config
        base_cfg = OmegaConf.create({
            "save_total_limit": 3,
            "checkpoint_retention": {
                "keep_every_n_steps": 10000,  # No checkpoints match this
                "keep_milestone_steps": [],   # No milestones
            }
        })

        trainer = self._create_minimal_trainer(base_cfg)

        # Call _rotate_checkpoints
        trainer._rotate_checkpoints(use_mtime=False, output_dir=self.test_dir)

        # Check that only 3 most recent remain
        remaining = self._get_checkpoint_steps()
        self.assertEqual(len(remaining), 3)
        self.assertEqual(remaining, [4000, 5000, 6000])

    def test_rotation_preserves_protected_directories(self):
        """Test that _rotate_checkpoints preserves protected directories."""
        # Create checkpoints
        steps = [1000, 5000, 10000, 15000, 20000, 25000, 30000]
        for step in steps:
            self._create_checkpoint(step)

        # Verify all exist
        self.assertEqual(self._get_checkpoint_steps(), steps)

        # Create trainer with retention config
        base_cfg = OmegaConf.create({
            "save_total_limit": 3,
            "checkpoint_retention": {
                "keep_every_n_steps": 10000,  # Protect 10000, 20000, 30000
                "keep_milestone_steps": [5000],  # Protect 5000
            }
        })

        trainer = self._create_minimal_trainer(base_cfg)

        # Call _rotate_checkpoints
        trainer._rotate_checkpoints(use_mtime=False, output_dir=self.test_dir)

        # Check that protected + 3 recent remain
        remaining = self._get_checkpoint_steps()
        expected = [5000, 10000, 20000, 25000, 30000]  # Protected + 3 recent
        self.assertEqual(sorted(remaining), expected)

    def test_rotation_with_best_model_checkpoint(self):
        """Test that _rotate_checkpoints preserves best_model_checkpoint."""
        # Create checkpoints
        steps = [1000, 5000, 10000, 15000]
        for step in steps:
            self._create_checkpoint(step)

        # Create trainer
        base_cfg = OmegaConf.create({
            "save_total_limit": 2,
            "checkpoint_retention": {},
        })

        trainer = self._create_minimal_trainer(base_cfg)

        # Set best_model_checkpoint to 5000
        trainer.state.best_model_checkpoint = str(self.test_dir / "checkpoint-5000")

        # Call _rotate_checkpoints
        trainer._rotate_checkpoints(use_mtime=False, output_dir=self.test_dir)

        # Check that best_model_checkpoint (5000) + 2 recent (10000, 15000) remain
        remaining = self._get_checkpoint_steps()
        self.assertIn(5000, remaining)
        self.assertIn(10000, remaining)
        self.assertIn(15000, remaining)
        self.assertNotIn(1000, remaining)  # Should be deleted

    def test_rotation_with_all_checkpoints_protected(self):
        """Test rotation when all checkpoints are protected."""
        # Create checkpoints at protected intervals
        steps = [10000, 20000, 30000, 40000]
        for step in steps:
            self._create_checkpoint(step)

        # Create trainer with retention config protecting all
        base_cfg = OmegaConf.create({
            "save_total_limit": 1,  # Would normally delete 3
            "checkpoint_retention": {
                "keep_every_n_steps": 10000,  # Protects all of them
            }
        })

        trainer = self._create_minimal_trainer(base_cfg)

        # Call _rotate_checkpoints
        trainer._rotate_checkpoints(use_mtime=False, output_dir=self.test_dir)

        # All should remain since they're all protected
        remaining = self._get_checkpoint_steps()
        self.assertEqual(sorted(remaining), steps)


if __name__ == "__main__":
    import unittest
    unittest.main()
