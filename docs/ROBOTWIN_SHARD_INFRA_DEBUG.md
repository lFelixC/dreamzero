# RoboTwin Shard Infra Debug

This note tracks the infra-facing data loading checks for RoboTwin Wan2.2 training.
The goal is to separate GPU/NCCL issues from data, shard, memory, and cache behavior.

## Checklist

- [x] Build a full offline shard manifest for all RoboTwin shards and write `shard_profile.jsonl`.
- [x] Replay selected shards offline and measure real `get_shard()` cache time, peak RSS, and per-view decode cost.
- [x] Add segmented timing around shard caching so training logs show parquet read, video decode, concat, and transform time.
- [x] Monitor memory, NUMA, page faults, and worker I/O during training to catch cache or host-memory pressure.
- [x] Redesign or rebalance shard scheduling by estimated cost instead of step count only.
- [x] Maintain a rank-wait versus data-wait runbook with HTML reports for every visualization.

## Output Locations

Default profiler outputs:

- JSONL: `/data/checkpoints/dreamzero/robotwin_shard_profile/shard_profile.jsonl`
- Summary JSON: `/data/checkpoints/dreamzero/robotwin_shard_profile/shard_profile.summary.json`
- HTML report: `/data/checkpoints/dreamzero/robotwin_shard_profile/shard_profile.html`
- Replay JSONL: `/data/checkpoints/dreamzero/robotwin_shard_profile/shard_replay_profile.jsonl`
- Replay summary JSON: `/data/checkpoints/dreamzero/robotwin_shard_profile/shard_replay_profile.summary.json`
- Training timing events JSONL: `/data/checkpoints/dreamzero/robotwin_shard_profile/shard_timing_events.jsonl`
- Training timing summary JSON: `/data/checkpoints/dreamzero/robotwin_shard_profile/shard_timing_events.summary.json`
- Training timing smoke log: `/data/checkpoints/dreamzero/robotwin_shard_profile/shard_timing_train_smoke.log`
- Host infra monitor JSONL: `/data/checkpoints/dreamzero/robotwin_shard_profile/shard_infra_monitor.jsonl`
- Host infra monitor summary JSON: `/data/checkpoints/dreamzero/robotwin_shard_profile/shard_infra_monitor.summary.json`
- Host infra monitor training log: `/data/checkpoints/dreamzero/robotwin_shard_profile/shard_infra_train_smoke.log`
- Shard schedule balance JSON: `/data/checkpoints/dreamzero/robotwin_shard_profile/shard_schedule_balance.json`
- Shard schedule balance smoke log: `/data/checkpoints/dreamzero/robotwin_shard_profile/shard_schedule_balance_smoke.log`
- Rank/data wait runbook: Step 6 in `/data/checkpoints/dreamzero/robotwin_shard_profile/shard_profile.html`

## Rank-Wait Versus Data-Wait Runbook

Use this when a specific GPU, for example card 2/3, shows unexpectedly low utilization.
The goal is to decide whether the rank is waiting for data/shard cache, waiting for another rank, or blocked in model/communication.

### Evidence to keep

- Training stdout/stderr with `DREAMZERO_SHARD_TIMING=1`.
- Parsed timing report appended to `/data/checkpoints/dreamzero/robotwin_shard_profile/shard_profile.html`.
- Host monitor JSONL and summary from `scripts/data/monitor_shard_infra.py`.
- The static shard cost profile at `/data/checkpoints/dreamzero/robotwin_shard_profile/shard_profile.jsonl`.
- The cost balance report at `/data/checkpoints/dreamzero/robotwin_shard_profile/shard_schedule_balance.json`.

### Decision table

| Observed signal | Likely meaning | Next action |
| --- | --- | --- |
| `shard_wait` is close to `shard_cache`, and video decode is most of cache time | GPU/rank is waiting for data cache | Check shard index/source bytes, enable `cost_grouped`, then reduce raw cache/decode pressure if wait remains high |
| One rank waits more and its shard has much higher `source_total_bytes` | Shard assignment imbalance is making faster ranks wait | Compare timing logs with `shard_schedule_balance.json`; enable cost-aware scheduling first |
| `shard_wait` is low but GPU utilization is still low | Not primarily shard cache; could be transform, collator, CPU, optimizer, or collectives | Inspect `transform_seconds_per_sample`; if low too, move to torch profiler/NCCL profiling |
| MemAvailable falls, major faults rise, or RSS peaks align with wait spikes | Host memory/page-cache pressure is affecting the data path | Reduce worker/prefetch/cache concurrency; inspect NUMA placement and page faults |
| All ranks are low-utilization together, data wait is low, and collective time is high | More likely rank/NCCL/model-side synchronization | Keep data timing as exclusion evidence, then debug topology/NCCL/optimizer |

### Runtime toggles

Enable data timing:

```bash
DREAMZERO_SHARD_TIMING=1
DREAMZERO_SHARD_TIMING_SAMPLE_INTERVAL=100
```

Enable cost-aware shard scheduling:

```bash
DREAMZERO_SHARD_SCHEDULE_BALANCE=cost_grouped
DREAMZERO_SHARD_COST_PROFILE=/data/checkpoints/dreamzero/robotwin_shard_profile/shard_profile.jsonl
DREAMZERO_SHARD_COST_KEY=source_total_bytes
```

### Current conclusion

Current evidence points first to data/cache/decode and shard-cost imbalance, not a pure NCCL issue:

- Offline replay: video decode dominates real `get_shard()` cache time.
- Training smoke: `shard_wait` is close to cache time, so the first shard is not hidden by prefetch.
- Host monitor: no extreme NUMA skew in the smoke, but host RSS and startup page faults are large enough to keep monitoring.
- Cost-grouped simulation: P90 round wait waste drops from 29.43 GiB to 3.07 MiB.

## Current Notes

- 2026-05-10: completed the static scan for 608 shards.
  - Validation: 608 JSONL rows, 0 missing video files, 0 missing parquet files, 0 shard step-count mismatches.
  - Largest source-byte shards: 71, 70, 72, 67, 66.
- 2026-05-10: completed segmented offline replay for representative shards 15, 96, 598, 71, and 190.
  - Replay was appended to the same HTML report, with Chinese legend explanations and recommendations.
  - Cache time range: 55.2s to 139.0s; slowest shard: 71.
  - Video decode dominates cache time: 45.4s to 128.4s; parquet read is only 0.1s to 0.5s.
  - Raw frame cache size is 25.5 GiB to 27.6 GiB per shard, with peak RSS delta 33.9 GiB to 36.7 GiB.
  - Source bytes and cache time are strongly correlated in this run: Pearson r = 0.99.
  - Recommendation: treat this as a data/cache/decode imbalance first. Avoid increasing dataloader workers or prefetch before reducing raw cache pressure or adding cost-aware shard scheduling.
- 2026-05-10: added structured shard timing logs in the training data path.
  - Log prefix: `DREAMZERO_SHARD_TIMING`.
  - Events: `shard_cache_submit`, `shard_cache`, `shard_wait`, `shard_samples_progress`, and `shard_samples`.
  - `shard_cache` includes parquet read, video decode by view, frame concat, dataframe concat, cache wall time, frame counts, decoded bytes, and worker RSS.
  - `shard_wait` includes rank, worker, schedule index, shard index, and `finish_cache_shard()` wait time.
  - `shard_samples_progress`/`shard_samples` include get-step-data and transform time per sample. The first sample of every shard is logged immediately; then progress logs default to every 100 yielded samples.
  - Controls: set `DREAMZERO_SHARD_TIMING=0` to disable, or `DREAMZERO_SHARD_TIMING_SAMPLE_INTERVAL=N` to change progress frequency.
  - Smoke validation: two-GPU training reached one step and emitted cache/wait timing; the save phase was stopped to avoid writing a large temporary full checkpoint. Iterator smoke emitted transform progress and final sample timing.
  - Parsed smoke summary: 29 timing events, cache mean 6.47s, video decode mean 5.59s, wait mean 6.19s, transform mean 0.38s/sample, get-step-data mean 0.025s/sample.
- 2026-05-10: added and ran host infra monitor for a two-GPU training smoke.
  - Monitor script: `scripts/data/monitor_shard_infra.py`.
  - It samples the torchrun process tree from `/proc`: RSS/HWM/RssAnon/RssFile/RssShmem, page faults, `/proc/<pid>/io`, NUMA page placement, process role, and system meminfo.
  - The same smoke also refreshed `DREAMZERO_SHARD_TIMING` so cache events now include rank, worker, dataset index, and schedule index instead of showing `rank ?`.
  - Smoke result: 157 samples over 280.3s; training reached one step successfully.
  - Peak process-tree RSS: 147.2 GiB at 148.6s. Peak single rank RSS: 63.9 GiB.
  - MemAvailable went from 1436.5 GiB to a minimum of 1317.2 GiB, then recovered to 1435.0 GiB.
  - Representative NUMA placement at RSS peak: N0 57.9 GiB, N1 89.6 GiB; largest node share 60.7%, so this smoke does not show extreme single-node NUMA skew.
  - Page-fault delta: 116.6M minor faults and 19.9K major faults. Because this smoke includes model/checkpoint loading, treat major faults as startup pressure unless they align with later shard wait spikes in a longer run.
  - Process write delta was 47.0 GiB because the framework still wrote final model files even with the short smoke; the temporary smoke output directory was deleted after validation.
- 2026-05-10: added cost-aware shard scheduling and an offline balance analysis.
  - Implementation: `ShardedLeRobotMixtureDataset` can now reorder the sampled shard schedule with `cost_grouped`.
  - Default behavior is unchanged. The feature is off unless `DREAMZERO_SHARD_SCHEDULE_BALANCE=cost_grouped` is set.
  - Runtime enable:
    - `DREAMZERO_SHARD_SCHEDULE_BALANCE=cost_grouped`
    - `DREAMZERO_SHARD_COST_PROFILE=/data/checkpoints/dreamzero/robotwin_shard_profile/shard_profile.jsonl`
    - `DREAMZERO_SHARD_COST_KEY=source_total_bytes`
  - The sampled shard multiset is preserved; only the order changes. The scheduler sorts by estimated cost, groups by `world_size * num_workers`, and shuffles groups during training so each round assigns similarly expensive shards to ranks/workers.
  - Offline 2-rank/1-worker simulation over 4096 sampled shards: P90 round wait waste dropped from 29.43 GiB to 3.07 MiB; mean round wait waste dropped from 13.36 GiB to 16.57 MiB.
  - Slot total cost ratio improved from 1.014 to 1.000 in the same simulation.
  - Runtime smoke loaded 608 shard costs and emitted a `DREAMZERO_SHARD_TIMING` `shard_schedule_balance` event without triggering video decode.
  - HTML report now has visible step dividers, so each appended visualization is labeled by investigation step.
- 2026-05-10: completed the rank-wait versus data-wait runbook.
  - Generator script: `scripts/data/append_robotwin_rank_data_runbook.py`.
  - Step divider script updated for Step 6: `scripts/data/annotate_shard_profile_html_steps.py`.
  - The shared HTML report now has Step 1 through Step 6 sections, including a decision matrix, evidence snapshot, operational commands, and recommendations.
  - Checklist completed; future training investigations should append new timing/monitor results to the same HTML and re-run the divider annotator.
- The first pass is a static manifest scan. It does not decode every video frame.
- Static profiling is intentionally cheap enough to run while training is active.
- Offline replay is measured under the current OS page-cache state. We intentionally did not drop caches because training was running.
