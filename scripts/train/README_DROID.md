# DROID Training Scripts

The canonical DROID Wan2.2 launchers are:

| Script | Purpose |
| --- | --- |
| `droid_wan22_joint_fseq200.sh` | Joint baseline, 320x640 composite, `frame_seqlen=200` |
| `droid_wan22_mot_fseq200.sh` | MoT baseline, 320x640 composite, `frame_seqlen=200` |
| `droid_wan22_fseq200.sh` | Shared implementation; use `ARCH=joint` or `ARCH=mot` |

Old DROID Wan2.2 script names are compatibility wrappers and route to these
200-token launchers. Avoid adding new experiments to the old names.

## Shared Defaults

Both Joint and MoT use the same data and video-token setup by default:

| Setting | Value |
| --- | --- |
| Data config | `data=dreamzero/droid_relative_wan22` |
| Backbone | `Wan2.2-TI2V-5B` |
| Train architecture | `full` |
| Input frames | `num_frames=33` |
| Action horizon | `action_horizon=24` |
| Views | `num_views=3` |
| Per-view resize | `image_resolution_height=160`, `image_resolution_width=320` |
| Final action-head video size | `target_video_height=320`, `target_video_width=640` |
| Frame tokens | `frame_seqlen=200` |
| Block alignment | `num_frame_per_block=2`, `num_action_per_block=24`, `num_state_per_block=1` |
| Default chunk limit | `max_chunk_size=4` |
| Default LR | `1e-5` |
| Default DeepSpeed | `zero2_offload` |

The 200-token setting comes from:

```text
320x640 video
  -> Wan2.2 VAE38 spatial downscale 16x: 20x40 latent
  -> DiT patch stride 2x2: 10x20 tokens
  -> frame_seqlen = 200
```

## Joint vs MoT Diff

| Setting | Joint | MoT |
| --- | --- | --- |
| `architecture` | `joint` | `mot` |
| Action head config | `wan_flow_matching_action_tf_wan22` | `wan_flow_matching_action_tf_wan22_mot` |
| Transformer layout | Shared video/action DiT | Wan video expert + separate action expert |
| Action-video attention | Not applicable | `mot_action_video_attention=full_video` |
| KI detach | Not applicable | `mot_action_video_ki=false` |
| MoT action hidden | Not applicable | `1024` |
| MoT action FFN | Not applicable | `4096` |
| MoT action heads | Not applicable | `8` |

Everything else is intentionally kept aligned so Joint vs MoT comparisons are
not confounded by resolution, token count, data sampling, or block alignment.

## Examples

Single-node Joint:

```bash
cd /data/dreamzero
CUDA_VISIBLE_DEVICES=4,5,6,7 \
OUTPUT_DIR=/data/checkpoints/dreamzero/droid_joint_fseq200 \
bash scripts/train/droid_wan22_joint_fseq200.sh
```

Single-node MoT:

```bash
cd /data/dreamzero
CUDA_VISIBLE_DEVICES=4,5,6,7 \
OUTPUT_DIR=/data/checkpoints/dreamzero/droid_mot_fseq200 \
bash scripts/train/droid_wan22_mot_fseq200.sh
```

Multi-node uses the same scripts. Run on each node with the same
`MASTER_ADDR`, `MASTER_PORT`, and `NNODES`, changing only `NODE_RANK`:

```bash
NNODES=2 NODE_RANK=0 MASTER_ADDR=<rank0_ip> bash scripts/train/droid_wan22_mot_fseq200.sh
NNODES=2 NODE_RANK=1 MASTER_ADDR=<rank0_ip> bash scripts/train/droid_wan22_mot_fseq200.sh
```

Useful overrides:

```bash
PER_DEVICE_BS=2
MAX_CHUNK_SIZE=4
DEEPSPEED_CFG=zero2
MAX_STEPS=100000
DROID_RANDOM_DROP_EXTERIOR_VIEW_PROB=0.15
```
