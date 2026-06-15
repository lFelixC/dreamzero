# Checkpoint Retention

This document describes the two-tier checkpoint retention system in DreamZero.

## Overview

The checkpoint retention system provides a flexible way to manage long-term checkpoint storage while respecting disk space constraints. It separates checkpoint management into two tiers:

1. **High-frequency temporary checkpoints** - For recent checkpoints used in training and short-term rollback
2. **Long-term retention** - For important checkpoints that should be kept indefinitely

## Configuration

Configure checkpoint retention in your training config:

```yaml
# Standard HuggingFace checkpoint settings
save_steps: 1000           # Save a checkpoint every 1000 steps
save_total_limit: 5        # Keep only 5 recent checkpoints for rotation

# Long-term retention rules (new)
checkpoint_retention:
  keep_every_n_steps: 20000        # Keep one checkpoint every 20000 steps
  keep_milestone_steps: [5000, 10000, 50000]  # Keep specific milestone checkpoints
  keep_best_n: 0                    # Keep N best checkpoints (future: requires eval metrics)
```

## Retention Rules

### `keep_every_n_steps`

Keeps checkpoints at regular intervals. Useful for keeping long-term snapshots of training progress.

**Example:**
```yaml
checkpoint_retention:
  keep_every_n_steps: 20000
```

At step 60000, this protects checkpoints at: 20000, 40000, 60000

### `keep_milestone_steps`

Keeps checkpoints at explicitly specified steps. Useful for keeping early training checkpoints or specific evaluation points.

**Example:**
```yaml
checkpoint_retention:
  keep_milestone_steps: [5000, 10000, 50000, 100000]
```

This protects checkpoints at steps 5000, 10000, 50000, and 100000 (if they exist).

### `keep_best_n` (Future)

Will keep the N best checkpoints based on evaluation metrics. This requires integration with evaluation metrics.

**Example (future):**
```yaml
checkpoint_retention:
  keep_best_n: 3
  metric_for_best_model: "eval/success_rate"
  mode: "max"  # or "min" for loss
```

## How It Works

1. **Temporary checkpoints** continue to work as before:
   - A checkpoint is saved every `save_steps` steps
   - Only `save_total_limit` recent non-protected checkpoints are kept

2. **Protected checkpoints** are excluded from rotation:
   - Checkpoints matching retention rules are never deleted
   - They count as "extra" beyond `save_total_limit`

## Example

With this configuration:
```yaml
save_steps: 1000
save_total_limit: 3
checkpoint_retention:
  keep_every_n_steps: 10000
  keep_milestone_steps: [5000]
```

At step 25000, with checkpoints at [1000, 2000, 3000, ..., 25000]:

- **Protected checkpoints (never deleted):** 5000 (milestone), 10000 (every_n), 20000 (every_n)
- **Recent checkpoints (up to 3):** 23000, 24000, 25000
- **Deleted checkpoints:** Everything else

## Implementation Details

The retention logic is implemented in:
- `groot/vla/experiment/checkpoint_retention.py` - Core retention logic
- `groot/vla/experiment/base.py` - BaseTrainer._rotate_checkpoints() override

The system works by:
1. Maintaining a set of "protected" checkpoint steps based on configured rules
2. When rotating checkpoints, skipping protected checkpoints from deletion
3. Deleting only the oldest non-protected checkpoints to maintain `save_total_limit`
