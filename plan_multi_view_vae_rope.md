# Token-Level Multi-View Input with View-Aware 3D RoPE 实现计划

## 需求概述

将当前的多视图输入处理从"VAE编码前拼接"改为"独立VAE编码+视图感知RoPE"。

### 核心需求

1. **独立VAE编码**：每个相机视图独立通过VAE编码
2. **独立Patchify**：每个视图的latent grid分别patchify
3. **后拼接**：在VAE编码和patchify之后才拼接tokens
4. **视图感知RoPE**：使用局部坐标 `pᵛ = (t, local_y + view_y_offset[v], local_x + view_x_offset[v])`
5. **可变视图数**：支持1-3个视图（推理时动态）
6. **边界检查**：验证跨视图tokens在RoPE空间不相邻
7. **向后兼容**：保留原有拼接方式用于对比实验

### 设计决策

| 项目 | 决策 |
|------|------|
| 位置映射 | 简单水平偏移（view_x_offset = view_id * W） |
| 兼容性 | 需要兼容，通过配置切换 |
| Token排序 | 时间优先：View1-T1, View2-T1, View3-T1, View1-T2, ... |

## 当前架构分析

### 当前流程
```
原始视图 (3个) → 拼接(concat.py/video.py) → VAE编码 → Patch Embedding → RoPE → Transformer
```

### 关键文件

| 文件 | 作用 |
|------|------|
| `groot/vla/data/transform/video.py` | 视图拼接逻辑 |
| `groot/vla/data/transform/concat.py` | 拼接顺序配置 |
| `groot/vla/model/dreamzero/modules/wan_video_vae.py` | VAE编码 |
| `groot/vla/model/dreamzero/modules/wan_video_dit.py` | RoPE实现 |
| `groot/vla/model/dreamzero/modules/wan2_1_submodule.py` | Patch embedding |

## 实现计划

### Phase 1: 添加新模块

#### 1.1 创建视图感知RoPE模块

**新建文件**: `groot/vla/model/dreamzero/modules/view_aware_rope.py`

```python
"""
View-aware RoPE for multi-view inputs.
"""
import torch
import torch.nn as nn
from typing import Dict, List

class ViewAwareRoPE:
    """View-aware 3D Rotary Position Embedding."""

    def __init__(self, dim: int, theta: float = 10000.0):
        self.dim = dim
        self.theta = theta

    def compute_view_offsets(
        self,
        view_ids: List[int],
        latent_width: int
    ) -> Dict[int, int]:
        """
        Compute horizontal offsets for each view.

        Args:
            view_ids: List of view IDs
            latent_width: Width of latent grid

        Returns:
            {view_id: x_offset} mapping
        """
        # Simple horizontal offset: each view shifted by latent_width
        offsets = {}
        for i, view_id in enumerate(view_ids):
            offsets[view_id] = i * latent_width
        return offsets

    def precompute_freqs_cis_3d(
        self,
        grid_size: tuple,
        view_offsets: Dict[int, int],
        max_seq_len: int = 4096
    ):
        """
        Precompute 3D RoPE frequencies with view offsets.

        Args:
            grid_size: (f, h, w) temporal/spatial sizes
            view_offsets: {view_id: x_offset}
            max_seq_len: Maximum sequence length

        Returns:
            Dict with frequency components
        """
        f, h, w = grid_size

        # Calculate max width including all view offsets
        max_x_offset = max(view_offsets.values()) if view_offsets else 0
        total_w = w + max_x_offset

        # Time dimension (no offset)
        dim_t = self.dim - 2 * (self.dim // 3)
        freqs_t = self._precompute_freqs_1d(dim_t, f, max_seq_len)

        # Height dimension (no offset for now)
        dim_h = self.dim // 3
        freqs_h = self._precompute_freqs_1d(dim_h, h, max_seq_len)

        # Width dimension (with view offsets)
        dim_w = self.dim // 3
        freqs_w = self._precompute_freqs_1d(dim_w, total_w, max_seq_len)

        return {
            "t": freqs_t,
            "h": freqs_h,
            "w": freqs_w,
            "view_offsets": view_offsets
        }

    def _precompute_freqs_1d(self, dim: int, end: int, max_end: int):
        """Precompute 1D frequency components."""
        # Standard RoPE frequency computation
        freqs = 1.0 / (self.theta ** (torch.arange(0, dim, 2)[:dim//2].float() / dim))
        t = torch.arange(end, max_end)
        freqs = torch.outer(t, freqs)
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs
```

#### 1.2 创建视图边界检查工具

**新建文件**: `groot/vla/model/dreamzero/utils/view_boundary_check.py`

```python
"""Utilities for checking view boundary separation."""

import torch
import logging

logger = logging.getLogger(__name__)

def check_view_boundary_separation(
    view_indices: torch.Tensor,
    positions: torch.Tensor,
    view_offsets: Dict[int, int],
    min_separation: int = 1
) -> bool:
    """
    Verify tokens from different views are separated in RoPE space.

    Args:
        view_indices: [N] view_id for each token
        positions: [N, 3] local (t, y, x) positions
        view_offsets: {view_id: x_offset}
        min_separation: Minimum required distance

    Returns:
        True if boundaries are properly separated
    """
    if len(positions) < 2:
        return True

    # Compute global positions with view offsets
    global_positions = positions.clone()
    for i, view_id in enumerate(view_indices):
        x_off = view_offsets.get(view_id.item(), 0)
        global_positions[i, 2] += x_off  # Add offset to x dimension

    # Check consecutive tokens from different views
    violations = []
    for i in range(len(global_positions) - 1):
        if view_indices[i] != view_indices[i + 1]:
            # Only check x distance since that's where we apply offset
            x_dist = abs(global_positions[i, 2] - global_positions[i + 1, 2])
            if x_dist < min_separation:
                violations.append((i, i + 1, x_dist))

    if violations:
        logger.warning(
            f"Found {len(violations)} view boundary violations: "
            f"{violations[:5]}..."  # Show first 5
        )
        return False

    return True

def compute_token_positions(
    grid_size: tuple,
    patch_size: tuple = (1, 2, 2)
) -> torch.Tensor:
    """
    Compute (t, y, x) positions for all tokens in a grid.

    Args:
        grid_size: (f, h, w) grid dimensions
        patch_size: (t_patch, h_patch, w_patch)

    Returns:
        [num_tokens, 3] position tensor
    """
    f, h, w = grid_size
    t_p, h_p, w_p = patch_size

    num_f = f // t_p
    num_h = h // h_p
    num_w = w // w_p

    positions = []
    for t in range(num_f):
        for y in range(num_h):
            for x in range(num_w):
                positions.append([t, y, x])

    return torch.tensor(positions, dtype=torch.long)
```

### Phase 2: 修改现有模块

#### 2.1 修改视频Transform（添加配置开关）

**修改文件**: `groot/vla/data/transform/video.py`

在 `VideoTransform` 类中添加配置选项：

```python
@dataclass
class VideoTransform:
    # ... 现有字段 ...

    # 新增字段
    use_independent_vae: bool = Field(
        default=False,
        description="Whether to encode views independently with VAE"
    )
    view_aware_rope: bool = Field(
        default=False,
        description="Whether to use view-aware RoPE"
    )

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.use_independent_vae:
            return self._apply_independent(data)
        else:
            return self._apply_stitched(data)  # 原有逻辑

    def _apply_independent(self, data: dict[str, Any]) -> dict[str, Any]:
        """New independent encoding path."""
        views = []
        view_metadata = []

        for view_id, key in enumerate(self.apply_to):
            if key not in data:
                continue
            video = data[key]
            views.append(video)
            view_metadata.append({
                'view_id': view_id,
                'key': key,
            })

        # Store as list instead of concatenating
        data['video_views'] = views
        data['view_metadata'] = view_metadata

        return data
```

#### 2.2 修改VAE编码接口

**修改文件**: `groot/vla/model/dreamzero/modules/wan_video_vae.py`

添加独立编码方法：

```python
class WanVideoVAE(nn.Module):
    # ... 现有代码 ...

    def encode_independent(
        self,
        videos_list: List[torch.Tensor],
        view_metadata: List[dict]
    ) -> tuple:
        """
        Encode each view independently.

        Args:
            videos_list: List of [B, T, C, H, W] videos
            view_metadata: List of metadata dicts

        Returns:
            latents_list: List of encoded latents
            updated_metadata: Metadata with latent info
        """
        latents_list = []
        updated_metadata = []

        for video, meta in zip(videos_list, view_metadata):
            # Encode using existing method
            latent = self.encode(video, tiled=self.tiled_decode)
            latents_list.append(latent)

            # Update metadata with latent shape
            meta['latent_shape'] = latent.shape  # [B, z_dim, h, w]
            updated_metadata.append(meta)

        return latents_list, updated_metadata
```

#### 2.3 修改Patch Embedding

**修改文件**: `groot/vla/model/dreamzero/modules/wan2_1_submodule.py`

添加视图感知的patch embedding：

```python
class WanVideoTransformer(nn.Module):
    def __init__(self, ...):
        # ... 现有代码 ...
        self.use_view_aware = False  # Config flag

    def patch_embedding_independent(
        self,
        latents_list: List[torch.Tensor],
        view_metadata: List[dict],
        patch_size: tuple = (1, 2, 2)
    ) -> tuple:
        """
        Create patch embeddings with view information.

        Returns:
            tokens: [B, total_tokens, C]
            view_indices: [total_tokens]
            positions: [total_tokens, 3] (t, y, x)
        """
        all_tokens = []
        all_view_indices = []
        all_positions = []

        for latent, meta in zip(latents_list, view_metadata):
            view_id = meta['view_id']

            # Apply patch embedding
            patches = self.patch_embedding(latent)
            b, c, f, h, w = patches.shape
            tokens = patches.flatten(2).transpose(1, 2)  # [B, F*H*W, C]

            all_tokens.append(tokens)

            # Record view_id for each token
            num_tokens = tokens.size(1)
            all_view_indices.extend([view_id] * num_tokens)

            # Compute positions
            for t_idx in range(f):
                for y_idx in range(h):
                    for x_idx in range(w):
                        all_positions.append([t_idx, y_idx, x_idx])

        return (
            torch.cat(all_tokens, dim=1),
            torch.tensor(all_view_indices),
            torch.tensor(all_positions)
        )
```

#### 2.4 修改RoPE应用

**修改文件**: `groot/vla/model/dreamzero/modules/wan_video_dit.py`

在现有的RoPE类中添加视图感知方法：

```python
class RotaryPositionEmbeddingNoPolarOp(nn.Module):
    # ... 现有代码 ...

    def forward_with_views(
        self,
        f: int,
        h: int,
        w: int,
        view_indices: torch.Tensor,
        view_offsets: Dict[int, int]
    ) -> torch.Tensor:
        """
        Apply RoPE with view-aware offsets.

        Args:
            f, h, w: Grid dimensions
            view_indices: [N] view_id per token
            view_offsets: {view_id: x_offset}

        Returns:
            RoPE frequencies for each token
        """
        # Get base frequencies
        freqs_cos_3d, freqs_sin_3d = self.forward(f, h, w, 0)

        # Apply view offsets to width component
        num_tokens = f * h * w
        offset_freqs_cos = freqs_cos_3d.clone()
        offset_freqs_sin = freqs_sin_3d.clone()

        # This is a simplified version - actual implementation needs
        # to handle the offset properly in the frequency computation
        return offset_freqs_cos, offset_freqs_sin
```

### Phase 3: 模型集成

#### 3.1 修改主模型入口

**修改文件**: `groot/vla/model/dreamzero/modules/dreamzero_mot.py`

在模型初始化和forward中添加视图感知支持：

```python
class DreamZeroMoT(nn.Module):
    def __init__(self, config, ...):
        # ... 现有代码 ...
        self.use_independent_vae = config.get('use_independent_vae', False)
        self.use_view_aware_rope = config.get('use_view_aware_rope', False)

        if self.use_view_aware_rope:
            from .view_aware_rope import ViewAwareRoPE
            self.view_aware_rope = ViewAwareRoPE(dim=config['hidden_dim'])

    def forward(self, x, view_metadata=None):
        if self.use_independent_vae and view_metadata is not None:
            return self._forward_independent(x, view_metadata)
        else:
            return self._forward_stitched(x)  # 原有逻辑

    def _forward_independent(self, x, view_metadata):
        """New forward path with independent encoding."""
        # 1. Independent VAE encoding
        latents_list, updated_meta = self.vae.encode_independent(x, view_metadata)

        # 2. Patch embedding with view info
        tokens, view_indices, positions = self.patch_embedding_independent(
            latents_list, updated_meta
        )

        # 3. Compute view offsets
        latent_width = latents_list[0].shape[-1]
        view_offsets = self.view_aware_rope.compute_view_offsets(
            [m['view_id'] for m in updated_meta],
            latent_width
        )

        # 4. Check boundaries
        from .utils.view_boundary_check import check_view_boundary_separation
        check_view_boundary_separation(view_indices, positions, view_offsets)

        # 5. Apply transformer with view-aware RoPE
        # ... 继续处理
```

### Phase 4: 训练和推理适配

#### 4.1 更新训练脚本

**修改文件**: `scripts/train/droid_wan22_mot_decoupled_full_video.sh`

添加配置选项：

```bash
# 在config中添加
use_independent_vae=true
use_view_aware_rope=true
```

#### 4.2 更新推理脚本

**修改文件**: `video_eval/server.py`, `socket_test_*.py`

确保推理时正确处理视图元数据和配置。

### Phase 5: 验证

#### 5.1 单元测试

新建 `tests/test_view_aware_rope.py`:

```python
def test_view_offset_computation():
    """Test horizontal offset calculation."""
    rope = ViewAwareRoPE(dim=72)
    offsets = rope.compute_view_offsets([0, 1, 2], latent_width=16)
    assert offsets == {0: 0, 1: 16, 2: 32}

def test_boundary_check():
    """Test view boundary detection."""
    view_indices = torch.tensor([0, 0, 0, 1, 1, 1])
    positions = torch.tensor([[0,0,0], [0,0,1], [0,0,2],
                              [0,0,0], [0,0,1], [0,0,2]])
    view_offsets = {0: 0, 1: 16}  # 16-pixel separation

    result = check_view_boundary_separation(view_indices, positions, view_offsets)
    assert result == True  # Should pass with sufficient separation
```

#### 5.2 对比实验

运行两组实验对比：
- 原始拼接方式
- 独立编码+视图感知RoPE

## 文件修改摘要

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `groot/vla/model/dreamzero/modules/view_aware_rope.py` | 新建 | 视图感知RoPE核心逻辑 |
| `groot/vla/model/dreamzero/utils/view_boundary_check.py` | 新建 | 边界检查工具 |
| `groot/vla/data/transform/video.py` | 修改 | 添加独立编码开关 |
| `groot/vla/model/dreamzero/modules/wan_video_vae.py` | 修改 | 添加独立编码方法 |
| `groot/vla/model/dreamzero/modules/wan2_1_submodule.py` | 修改 | 视图感知patch embedding |
| `groot/vla/model/dreamzero/modules/wan_video_dit.py` | 修改 | 视图感知RoPE应用 |
| `groot/vla/model/dreamzero/modules/dreamzero_mot.py` | 修改 | 模型入口集成 |
| `tests/test_view_aware_rope.py` | 新建 | 单元测试 |

## 预期问题和解决方案

### 问题1: 性能开销
**解决**: Batch并行VAE编码，缓存静态视图结果

### 问题2: 训练/推理视图数不一致
**解决**: RoPE频率基于最大视图数计算，推理时复用

### 问题3: 向后兼容性
**解决**: 使用配置开关，两种方式共存

### 问题4: 边界检查的最佳间隔
**解决**: 从保守值开始，通过消融实验确定

## 下一步

1. 实现Phase 1（新建模块）
2. 实现Phase 2-3（修改现有模块）
3. 实现Phase 4（训练/推理适配）
4. 实现Phase 5（验证和对比实验）
