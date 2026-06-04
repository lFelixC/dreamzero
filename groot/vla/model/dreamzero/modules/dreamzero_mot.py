from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import register_to_config

from groot.vla.model.dreamzero.modules.wan2_1_submodule import (
    WanRMSNorm,
    WanLayerNorm,
    rope_action_apply,
    sinusoidal_embedding_1d,
)
from groot.vla.model.dreamzero.modules.wan2_1_attention import AttentionModule
from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
    CategorySpecificMLP,
    CausalWanModel,
    MultiEmbodimentActionEncoder,
)

MoTActionVideoAttention = Literal["first_frame", "full_video", "full_video_unidirectional", "none"]
_MOT_ACTION_VIDEO_ATTENTION_MODES = ("first_frame", "full_video", "full_video_unidirectional", "none")
_MOT_ACTION_FULL_VIDEO_MODES = ("full_video", "full_video_unidirectional")


def _coerce_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("1", "true", "yes", "y", "on"):
            return True
        if normalized in ("0", "false", "no", "n", "off"):
            return False
    raise ValueError(f"Expected a boolean value, got {value!r}")


def _normalize_rope_freqs(freqs: torch.Tensor, *, complex_freqs: bool) -> torch.Tensor:
    if complex_freqs:
        if freqs.ndim == 2:
            return freqs.unsqueeze(1)
        return freqs
    if freqs.ndim == 3 and freqs.shape[1] == 1:
        return freqs.squeeze(1)
    return freqs


def _ordered_state_action_freqs(
    freqs_action: torch.Tensor,
    freqs_state: torch.Tensor,
    action_register_length: int,
    num_action_per_block: int,
    num_state_per_block: int,
    *,
    action_offset: int = 0,
    state_offset: int = 0,
) -> torch.Tensor:
    if num_state_per_block > 0:
        per_block = num_action_per_block + num_state_per_block
        if action_register_length % per_block == 0:
            chunk_size = action_register_length // per_block
        elif action_register_length % num_action_per_block == 0:
            chunk_size = action_register_length // num_action_per_block
            num_state_per_block = 0
        else:
            raise ValueError(
                "MoT action expert expects registers grouped as state+action tokens: "
                f"action_register_length={action_register_length}, "
                f"num_action_per_block={num_action_per_block}, "
                f"num_state_per_block={num_state_per_block}."
            )
    else:
        if action_register_length % num_action_per_block != 0:
            raise ValueError(
                "MoT action expert expects action registers to be divisible by "
                f"num_action_per_block={num_action_per_block}, got {action_register_length}."
            )
        chunk_size = action_register_length // num_action_per_block

    action_len = chunk_size * num_action_per_block
    state_len = chunk_size * num_state_per_block
    parts = []
    if state_len > 0:
        parts.append(freqs_state[state_offset : state_offset + state_len])
    parts.append(freqs_action[action_offset : action_offset + action_len])
    return torch.cat(parts, dim=0)


def _apply_rope_with_ordered_freqs(
    x: torch.Tensor,
    freqs: torch.Tensor,
    ordered_freqs: torch.Tensor,
) -> torch.Tensor:
    complex_freqs = torch.is_complex(ordered_freqs)
    freqs = _normalize_rope_freqs(freqs, complex_freqs=complex_freqs)
    ordered_freqs = _normalize_rope_freqs(ordered_freqs, complex_freqs=complex_freqs)
    freqs = torch.cat([freqs, ordered_freqs], dim=0)

    if complex_freqs:
        b, seq_len, n, _ = x.shape
        x_complex = torch.view_as_complex(x.to(torch.float64).reshape(b, seq_len, n, -1, 2))
        return torch.view_as_real(x_complex * freqs.unsqueeze(0)).flatten(3)

    freqs = freqs.unsqueeze(0).unsqueeze(2)
    x0, x1 = x.chunk(2, dim=-1)
    freqs_cos, freqs_sin = freqs.chunk(2, dim=-1)
    return torch.cat(
        (
            x0 * freqs_cos - x1 * freqs_sin,
            x1 * freqs_cos + x0 * freqs_sin,
        ),
        dim=-1,
    )


class DreamZeroActionExpertBlock(nn.Module):
    """Action expert block whose self-attention runs in Wan head space."""

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        num_heads: int,
        wan_num_heads: int,
        wan_head_dim: int,
        eps: float = 1e-6,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.wan_num_heads = wan_num_heads
        self.wan_head_dim = wan_head_dim
        self.wan_dim = wan_num_heads * wan_head_dim
        self.norm1 = WanLayerNorm(hidden_dim, eps)
        self.q = nn.Linear(hidden_dim, self.wan_dim)
        self.k = nn.Linear(hidden_dim, self.wan_dim)
        self.v = nn.Linear(hidden_dim, self.wan_dim)
        self.o = nn.Linear(self.wan_dim, hidden_dim)
        self.norm_q = WanRMSNorm(self.wan_dim, eps=eps)
        self.norm_k = WanRMSNorm(self.wan_dim, eps=eps)
        self.attn = AttentionModule(num_heads=wan_num_heads, head_dim=wan_head_dim)
        self.norm3 = WanLayerNorm(hidden_dim, eps)
        self.action_to_video_cross_proj = nn.Linear(hidden_dim, self.wan_dim)
        self.video_to_action_cross_proj = nn.Linear(self.wan_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = WanLayerNorm(hidden_dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, hidden_dim),
        )
        # AdaLN-zero style initialization keeps the randomly initialized action expert
        # numerically quiet at step 0, then lets the gates learn useful residuals.
        self.modulation = nn.Parameter(torch.zeros(1, 6, hidden_dim))
        self.cross_gate = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    @staticmethod
    def _align_modulation(parts: tuple[torch.Tensor, ...], length: int) -> tuple[torch.Tensor, ...]:
        aligned = []
        for part in parts:
            part = part.squeeze(2)
            if part.shape[1] == length:
                aligned.append(part)
            elif part.shape[1] > length:
                aligned.append(part[:, :length])
            else:
                repeat = (length + part.shape[1] - 1) // part.shape[1]
                aligned.append(part.repeat_interleave(repeat, dim=1)[:, :length])
        return tuple(aligned)

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        context: torch.Tensor | None,
    ) -> torch.Tensor:
        q, k, v, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.build_mixed_attention_io(
            x=x,
            e=e,
            freqs_action=None,
            freqs_state=None,
            action_register_length=None,
            num_action_per_block=None,
            num_state_per_block=None,
        )
        y = self.attn(q, k, v).flatten(2)
        x = self.apply_mixed_attention_output(
            residual_x=residual_x,
            mixed_attn_out=y,
            gate_msa=gate_msa,
            shift_mlp=shift_mlp,
            scale_mlp=scale_mlp,
            gate_mlp=gate_mlp,
            context=context,
        )
        return x

    def build_mixed_attention_io(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        freqs_action: torch.Tensor | None,
        freqs_state: torch.Tensor | None,
        action_register_length: int | None,
        num_action_per_block: int | None,
        num_state_per_block: int | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._align_modulation(
            (self.modulation.unsqueeze(1).to(dtype=e.dtype, device=e.device) + e).chunk(6, dim=2),
            x.shape[1],
        )

        y = self.norm1(x) * (1 + scale_msa) + shift_msa
        b, s = y.shape[:2]
        q = self.norm_q(self.q(y)).view(b, s, self.wan_num_heads, self.wan_head_dim)
        k = self.norm_k(self.k(y)).view(b, s, self.wan_num_heads, self.wan_head_dim)
        v = self.v(y).view(b, s, self.wan_num_heads, self.wan_head_dim)

        if action_register_length is not None:
            assert freqs_action is not None
            assert freqs_state is not None
            assert num_action_per_block is not None
            assert num_state_per_block is not None
            empty_video_freqs = freqs_action.new_empty((0, 1, freqs_action.shape[-1]))
            ordered_freqs = _ordered_state_action_freqs(
                freqs_action=freqs_action,
                freqs_state=freqs_state,
                action_register_length=action_register_length,
                num_action_per_block=num_action_per_block,
                num_state_per_block=num_state_per_block,
            )
            q = _apply_rope_with_ordered_freqs(q, empty_video_freqs, ordered_freqs).type_as(v)
            k = _apply_rope_with_ordered_freqs(k, empty_video_freqs, ordered_freqs).type_as(v)

        return q, k, v, x, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def apply_mixed_attention_output(
        self,
        residual_x: torch.Tensor,
        mixed_attn_out: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        context: torch.Tensor | None,
    ) -> torch.Tensor:
        y = torch.nan_to_num(self.o(mixed_attn_out))
        x = residual_x + y * gate_msa
        if context is not None and context.shape[1] > 0:
            y = self.norm3(x)
            y, _ = self.cross_attn(y, context, context, need_weights=False)
            y = torch.nan_to_num(y)
            x = x + y * self.cross_gate.to(dtype=y.dtype, device=y.device)

        y = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        x = x + torch.nan_to_num(self.ffn(y)) * gate_mlp
        return x


class DreamZeroActionExpert(nn.Module):
    """Independent action denoising expert for the DreamZero MoT WAM."""

    def __init__(
        self,
        action_dim: int,
        max_state_dim: int,
        hidden_dim: int,
        ffn_dim: int,
        text_dim: int,
        video_dim: int,
        freq_dim: int,
        num_layers: int,
        num_heads: int,
        wan_num_heads: int,
        wan_head_dim: int,
        eps: float = 1e-6,
        max_num_embodiments: int = 1,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.max_state_dim = max_state_dim
        self.hidden_dim = hidden_dim
        self.freq_dim = freq_dim
        self.num_layers = num_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=action_dim,
            hidden_size=hidden_dim,
            num_embodiments=max_num_embodiments,
        )
        self.state_encoder = CategorySpecificMLP(
            num_categories=max_num_embodiments,
            input_dim=max_state_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=max_num_embodiments,
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )
        self.text_context_proj = nn.Linear(text_dim, hidden_dim)
        self.context_norm = WanLayerNorm(hidden_dim, eps)
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.blocks = nn.ModuleList(
            [
                DreamZeroActionExpertBlock(
                    hidden_dim=hidden_dim,
                    ffn_dim=ffn_dim,
                    num_heads=num_heads,
                    wan_num_heads=wan_num_heads,
                    wan_head_dim=wan_head_dim,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )
        nn.init.zeros_(self.time_projection[-1].weight)
        nn.init.zeros_(self.time_projection[-1].bias)

    def _build_tokens(
        self,
        action: torch.Tensor,
        timestep_action: torch.Tensor,
        state: torch.Tensor | None,
        embodiment_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        action_tokens = self.action_encoder(action, timestep_action, embodiment_id)
        action_length = action_tokens.shape[1]
        if state is None:
            return action_tokens, timestep_action, 0, action_length

        state_tokens = self.state_encoder(state, embodiment_id)
        state_length = state_tokens.shape[1]
        if state_length == 0:
            return action_tokens, timestep_action, 0, action_length

        if timestep_action.shape[1] % state_length == 0:
            stride = timestep_action.shape[1] // state_length
            timestep_state = timestep_action[:, ::stride]
        else:
            indices = torch.linspace(
                0,
                timestep_action.shape[1] - 1,
                state_length,
                device=timestep_action.device,
                dtype=torch.long,
            )
            timestep_state = timestep_action[:, indices]

        tokens = torch.cat([state_tokens, action_tokens], dim=1)
        timestep_tokens = torch.cat([timestep_state, timestep_action], dim=1)
        return tokens, timestep_tokens, state_length, action_length

    def forward(
        self,
        action: torch.Tensor,
        timestep_action: torch.Tensor,
        state: torch.Tensor | None,
        embodiment_id: torch.Tensor,
        text_context: torch.Tensor,
    ) -> torch.Tensor:
        tokens, e, action_start, action_length = self.build_action_inputs(
            action=action,
            timestep_action=timestep_action,
            state=state,
            embodiment_id=embodiment_id,
        )
        context = self.build_context(text_context=text_context)

        for block in self.blocks:
            if torch.is_grad_enabled() and self.use_gradient_checkpointing:
                tokens = torch.utils.checkpoint.checkpoint(
                    block,
                    tokens,
                    e,
                    context,
                    use_reentrant=False,
                )
            else:
                tokens = block(tokens, e, context)

        action_tokens = tokens[:, action_start : action_start + action_length]
        return self.action_decoder(action_tokens, embodiment_id)

    def build_action_inputs(
        self,
        action: torch.Tensor,
        timestep_action: torch.Tensor,
        state: torch.Tensor | None,
        embodiment_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        tokens, timestep_tokens, action_start, action_length = self._build_tokens(
            action=action,
            timestep_action=timestep_action,
            state=state,
            embodiment_id=embodiment_id,
        )
        tokens = torch.nan_to_num(tokens)
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep_tokens.flatten()).type_as(tokens)
        )
        e = e.unflatten(dim=0, sizes=(tokens.shape[0], -1))
        e = self.time_projection(e).unflatten(dim=2, sizes=(6, self.hidden_dim))
        return tokens, torch.nan_to_num(e), action_start, action_length

    def build_context(
        self,
        text_context: torch.Tensor,
    ) -> torch.Tensor:
        context_parts = [self.text_context_proj(text_context)]
        return torch.nan_to_num(self.context_norm(torch.cat(context_parts, dim=1)))

    def decode_action(
        self,
        tokens: torch.Tensor,
        action_start: int,
        action_length: int,
        embodiment_id: torch.Tensor,
    ) -> torch.Tensor:
        return self.action_decoder(tokens[:, action_start : action_start + action_length], embodiment_id)


class MoTCausalWanModel(CausalWanModel):
    """DreamZero MoT WAM: Wan2.2 video expert plus an independent action expert."""

    @register_to_config
    def __init__(
        self,
        *args,
        mot_action_hidden_dim: int = 1024,
        mot_action_ffn_dim: int | None = None,
        mot_action_num_layers: int | None = None,
        mot_action_num_heads: int = 8,
        mot_action_video_attention: MoTActionVideoAttention = "full_video",
        mot_action_video_ki: bool | str = False,
        **kwargs,
    ):
        for removed_key in (
            "action_use_shared_context",
            "video_use_state_context",
            "detach_video_from_action_loss",
            "action_cross_gate_init",
            "action_gate_msa_init",
            "action_gate_mlp_init",
        ):
            kwargs.pop(f"mot_{removed_key}", None)
        super().__init__(*args, **kwargs)
        self.is_mot_wam = True
        if mot_action_video_attention == "causal":
            raise ValueError(
                "mot_action_video_attention='causal' was removed; use 'full_video' for block-causal "
                "video/action mixed attention, use 'full_video_unidirectional' for full video-to-action "
                "attention, or choose 'first_frame'/'none'."
            )
        if mot_action_video_attention not in _MOT_ACTION_VIDEO_ATTENTION_MODES:
            raise ValueError(f"Unsupported mot_action_video_attention={mot_action_video_attention!r}")
        self.mot_action_video_attention = mot_action_video_attention
        self.mot_action_video_ki = _coerce_bool(mot_action_video_ki)
        self.mot_action_reads_full_video = mot_action_video_attention in _MOT_ACTION_FULL_VIDEO_MODES
        self.mot_video_reads_action = mot_action_video_attention == "full_video"
        action_hidden_dim = int(mot_action_hidden_dim)
        action_ffn_dim = int(mot_action_ffn_dim or action_hidden_dim * 4)
        action_num_layers = int(mot_action_num_layers or self.num_layers)
        if action_num_layers != self.num_layers:
            raise ValueError(
                "Pi0-style mixed attention requires the action expert depth to match the Wan depth: "
                f"got action_layers={action_num_layers}, wan_layers={self.num_layers}."
            )
        wan_head_dim = self.dim // self.num_heads

        # The inherited joint action modules are intentionally not used by MoT. Deleting them
        # avoids shape conflicts when loading an old Joint checkpoint with strict=False.
        del self.action_encoder
        del self.state_encoder
        del self.action_decoder

        self.action_expert = DreamZeroActionExpert(
            action_dim=self.action_dim,
            max_state_dim=self.max_state_dim,
            hidden_dim=action_hidden_dim,
            ffn_dim=action_ffn_dim,
            text_dim=self.dim,
            video_dim=self.dim,
            freq_dim=self.freq_dim,
            num_layers=action_num_layers,
            num_heads=int(mot_action_num_heads),
            wan_num_heads=self.num_heads,
            wan_head_dim=wan_head_dim,
            eps=self.eps,
            max_num_embodiments=1,
            use_gradient_checkpointing=self.gradient_checkpointing,
        )
        self.state_context_encoder = CategorySpecificMLP(
            num_categories=1,
            input_dim=self.max_state_dim,
            hidden_dim=action_hidden_dim,
            output_dim=self.dim,
        )
        self.state_context_norm = WanLayerNorm(self.dim, self.eps)

    def _build_text_image_context(
        self,
        context: torch.Tensor,
        clip_feature: torch.Tensor | None,
    ) -> torch.Tensor:
        text_context = self.text_embedding(context)
        if clip_feature is not None:
            clip_embedding = self.img_emb(clip_feature)
            text_context = torch.cat([clip_embedding, text_context], dim=1)
        return text_context

    def _build_state_context(
        self,
        state: torch.Tensor | None,
        embodiment_id: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if state is None:
            return None
        # MoT currently uses a single shared state-context encoder category.
        embodiment_id = torch.zeros(batch_size, device=device, dtype=torch.long)
        state_context = self.state_context_encoder(state, embodiment_id)
        state_context = torch.nan_to_num(self.state_context_norm(state_context))
        return state_context

    def _merge_shared_context(
        self,
        text_image_context: torch.Tensor,
        state_context: torch.Tensor | None,
    ) -> torch.Tensor:
        if state_context is None:
            return text_image_context
        state_context = state_context.to(dtype=text_image_context.dtype, device=text_image_context.device)
        return torch.cat([state_context, text_image_context], dim=1)

    def _build_video_context(
        self,
        text_image_context: torch.Tensor,
        state_context: torch.Tensor | None,
    ) -> torch.Tensor:
        return text_image_context

    @staticmethod
    def _align_wan_modulation(
        block,
        e: torch.Tensor,
        length: int,
    ) -> tuple[torch.Tensor, ...]:
        parts = (block.modulation.unsqueeze(1).to(dtype=e.dtype, device=e.device) + e).chunk(6, dim=2)
        aligned = []
        for part in parts:
            part = part.squeeze(2)
            if part.shape[1] == length:
                aligned.append(part)
            elif part.shape[1] > length:
                aligned.append(part[:, :length])
            else:
                repeat = (length + part.shape[1] - 1) // part.shape[1]
                aligned.append(part.repeat_interleave(repeat, dim=1)[:, :length])
        return tuple(aligned)

    @staticmethod
    def _drop_full_true_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
        if mask is None:
            return None
        if bool(mask.all().item()):
            return None
        return mask

    @staticmethod
    def _align_token_mask(
        mask: torch.Tensor | None,
        length: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if mask is None:
            return None
        mask = mask.to(device=device, dtype=torch.bool)
        if mask.shape[1] == length:
            return mask
        if mask.shape[1] > length:
            return mask[:, :length]
        pad = torch.zeros(mask.shape[0], length - mask.shape[1], dtype=torch.bool, device=device)
        return torch.cat([mask, pad], dim=1)

    def _build_video_token_mask(
        self,
        video_frame_mask: torch.Tensor | None,
        video_frames: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if video_frame_mask is None:
            return None
        if seq_len % video_frames != 0:
            raise ValueError(f"seq_len={seq_len} is not divisible by video_frames={video_frames}")
        frame_mask = self._align_token_mask(video_frame_mask, video_frames, device)
        frame_mask = self._drop_full_true_mask(frame_mask)
        if frame_mask is None:
            return None
        return frame_mask.repeat_interleave(seq_len // video_frames, dim=1)

    def _concat_key_masks(
        self,
        masks: list[torch.Tensor | None],
        lengths: list[int],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if not any(mask is not None for mask in masks):
            return None
        aligned = []
        for mask, length in zip(masks, lengths):
            if mask is None:
                aligned.append(torch.ones(batch_size, length, dtype=torch.bool, device=device))
            else:
                aligned_mask = self._align_token_mask(mask, length, device)
                assert aligned_mask is not None
                aligned.append(aligned_mask)
        return self._drop_full_true_mask(torch.cat(aligned, dim=1))

    def _clean_image_attention_with_mask(
        self,
        self_attn,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_mask: torch.Tensor,
    ) -> torch.Tensor:
        block_size = self.frame_seqlen * self.num_frame_per_block
        total_len = q.shape[1]
        output = torch.empty_like(q)

        first_end = min(self.frame_seqlen, total_len)
        output[:, :first_end] = self_attn.attn(
            q[:, :first_end],
            k[:, :first_end],
            v[:, :first_end],
            key_mask=key_mask[:, :first_end],
        )

        for block_start in range(self.frame_seqlen, total_len, block_size):
            block_end = min(block_start + block_size, total_len)
            output[:, block_start:block_end] = self_attn.attn(
                q[:, block_start:block_end],
                k[:, :block_end],
                v[:, :block_end],
                key_mask=key_mask[:, :block_end],
            )

        return output

    def _noisy_image_attention_with_mask(
        self,
        self_attn,
        noisy_q: torch.Tensor,
        noisy_k: torch.Tensor,
        noisy_v: torch.Tensor,
        clean_k: torch.Tensor,
        clean_v: torch.Tensor,
        noisy_key_mask: torch.Tensor,
        clean_key_mask: torch.Tensor,
    ) -> torch.Tensor:
        block_size = self.frame_seqlen * self.num_frame_per_block
        total_len = noisy_q.shape[1]
        output = torch.empty_like(noisy_q)

        first_end = min(self.frame_seqlen, total_len)
        output[:, :first_end] = self_attn.attn(
            noisy_q[:, :first_end],
            noisy_k[:, :first_end],
            noisy_v[:, :first_end],
            key_mask=noisy_key_mask[:, :first_end],
        )

        for block_start in range(self.frame_seqlen, total_len, block_size):
            block_end = min(block_start + block_size, total_len)
            clean_end = min(block_start, clean_k.shape[1])
            k_context = torch.cat(
                [
                    clean_k[:, :clean_end],
                    noisy_k[:, block_start:block_end],
                ],
                dim=1,
            )
            v_context = torch.cat(
                [
                    clean_v[:, :clean_end],
                    noisy_v[:, block_start:block_end],
                ],
                dim=1,
            )
            key_context_mask = torch.cat(
                [
                    clean_key_mask[:, :clean_end],
                    noisy_key_mask[:, block_start:block_end],
                ],
                dim=1,
            )
            output[:, block_start:block_end] = self_attn.attn(
                noisy_q[:, block_start:block_end],
                k_context,
                v_context,
                key_mask=key_context_mask,
            )

        return output

    def _video_blockwise_attention_with_mask(
        self,
        self_attn,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_mask: torch.Tensor,
    ) -> torch.Tensor:
        total_len = q.shape[1]
        num_frames = total_len // self.frame_seqlen
        block_size = self.frame_seqlen * self.num_frame_per_block
        num_blocks = (num_frames - 1) // self.num_frame_per_block

        if num_blocks <= 0:
            return self_attn.attn(q, k, v, key_mask=key_mask)
        if self.local_attn_size == -1:
            return self_attn.causal_attn(q, k, v, key_mask=key_mask)

        output = torch.empty_like(q)
        first_end = min(self.frame_seqlen, total_len)
        output[:, :first_end] = self_attn.attn(
            q[:, :first_end],
            k[:, :first_end],
            v[:, :first_end],
            key_mask=key_mask[:, :first_end],
        )
        for block_start in range(self.frame_seqlen, total_len, block_size):
            block_end = min(block_start + block_size, total_len)
            kv_start = max(0, block_end - self.local_attn_size * self.frame_seqlen)
            output[:, block_start:block_end] = self_attn.attn(
                q[:, block_start:block_end],
                k[:, kv_start:block_end],
                v[:, kv_start:block_end],
                key_mask=key_mask[:, kv_start:block_end],
            )

        return output

    def _video_self_attention_with_kv(
        self,
        self_attn,
        x: torch.Tensor,
        freqs: torch.Tensor,
        is_tf: bool,
        key_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, s = x.shape[:2]
        n = self_attn.num_heads
        d = self_attn.head_dim
        q = self_attn.norm_q(self_attn.q(x)).view(b, s, n, d)
        k = self_attn.norm_k(self_attn.k(x)).view(b, s, n, d)
        v = self_attn.v(x).view(b, s, n, d)

        if is_tf:
            half_seq_len = s // 2
            q_clean = q[:, :half_seq_len]
            k_clean = k[:, :half_seq_len]
            q_noisy = q[:, half_seq_len:]
            k_noisy = k[:, half_seq_len:]

            rq_clean = rope_action_apply(
                x=q_clean,
                freqs=freqs,
                freqs_action=self.freqs_action,
                freqs_state=self.freqs_state,
                action_register_length=None,
            ).type_as(v)
            rk_clean = rope_action_apply(
                x=k_clean,
                freqs=freqs,
                freqs_action=self.freqs_action,
                freqs_state=self.freqs_state,
                action_register_length=None,
            ).type_as(v)
            rq_noisy = rope_action_apply(
                x=q_noisy,
                freqs=freqs,
                freqs_action=self.freqs_action,
                freqs_state=self.freqs_state,
                action_register_length=None,
            ).type_as(v)
            rk_noisy = rope_action_apply(
                x=k_noisy,
                freqs=freqs,
                freqs_action=self.freqs_action,
                freqs_state=self.freqs_state,
                action_register_length=None,
            ).type_as(v)

            clean_v = v[:, :half_seq_len]
            noisy_v = v[:, half_seq_len:]
            if key_mask is None:
                x_clean = self_attn._process_clean_image_only_stable(rq_clean, rk_clean, clean_v)
                x_noisy = self_attn._process_noisy_image_only_blocks(
                    rq_noisy,
                    rk_noisy,
                    noisy_v,
                    rk_clean,
                    clean_v,
                )
            else:
                clean_key_mask = key_mask[:, :half_seq_len]
                noisy_key_mask = key_mask[:, half_seq_len:]
                x_clean = self._clean_image_attention_with_mask(
                    self_attn,
                    rq_clean,
                    rk_clean,
                    clean_v,
                    clean_key_mask,
                )
                x_noisy = self._noisy_image_attention_with_mask(
                    self_attn,
                    rq_noisy,
                    rk_noisy,
                    noisy_v,
                    rk_clean,
                    clean_v,
                    noisy_key_mask,
                    clean_key_mask,
                )
            y = torch.cat([x_clean, x_noisy], dim=1)
            roped_k = torch.cat([rk_clean, rk_noisy], dim=1)
        else:
            roped_q = rope_action_apply(
                x=q,
                freqs=freqs,
                freqs_action=self.freqs_action,
                freqs_state=self.freqs_state,
                action_register_length=None,
            ).type_as(v)
            roped_k = rope_action_apply(
                x=k,
                freqs=freqs,
                freqs_action=self.freqs_action,
                freqs_state=self.freqs_state,
                action_register_length=None,
            ).type_as(v)
            if key_mask is None:
                y = self_attn._blockwise_causal_flash_attn(
                    roped_q,
                    roped_k,
                    v,
                    self.frame_seqlen,
                    self.num_frame_per_block,
                    action_horizon=None,
                    state_horizon=None,
                    num_action_per_block=None,
                    num_state_per_block=None,
                    visualize_mask=False,
                )
            else:
                y = self._video_blockwise_attention_with_mask(
                    self_attn,
                    roped_q,
                    roped_k,
                    v,
                    key_mask,
                )

        return y.flatten(2), roped_k, v

    def _build_video_mixed_attention_io(
        self,
        block,
        x: torch.Tensor,
        e: torch.Tensor,
        freqs: torch.Tensor,
        is_tf: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._align_wan_modulation(
            block,
            e,
            x.shape[1],
        )
        attn_input = block.norm1(x) * (1 + scale_msa) + shift_msa
        b, s = attn_input.shape[:2]
        n = block.self_attn.num_heads
        d = block.self_attn.head_dim
        q = block.self_attn.norm_q(block.self_attn.q(attn_input)).view(b, s, n, d)
        k = block.self_attn.norm_k(block.self_attn.k(attn_input)).view(b, s, n, d)
        v = block.self_attn.v(attn_input).view(b, s, n, d)

        if is_tf:
            half_seq_len = s // 2
            q_clean = rope_action_apply(
                x=q[:, :half_seq_len],
                freqs=freqs,
                freqs_action=self.freqs_action,
                freqs_state=self.freqs_state,
                action_register_length=None,
            ).type_as(v)
            k_clean = rope_action_apply(
                x=k[:, :half_seq_len],
                freqs=freqs,
                freqs_action=self.freqs_action,
                freqs_state=self.freqs_state,
                action_register_length=None,
            ).type_as(v)
            q_noisy = rope_action_apply(
                x=q[:, half_seq_len:],
                freqs=freqs,
                freqs_action=self.freqs_action,
                freqs_state=self.freqs_state,
                action_register_length=None,
            ).type_as(v)
            k_noisy = rope_action_apply(
                x=k[:, half_seq_len:],
                freqs=freqs,
                freqs_action=self.freqs_action,
                freqs_state=self.freqs_state,
                action_register_length=None,
            ).type_as(v)
            q = torch.cat([q_clean, q_noisy], dim=1)
            k = torch.cat([k_clean, k_noisy], dim=1)
        else:
            q = rope_action_apply(
                x=q,
                freqs=freqs,
                freqs_action=self.freqs_action,
                freqs_state=self.freqs_state,
                action_register_length=None,
            ).type_as(v)
            k = rope_action_apply(
                x=k,
                freqs=freqs,
                freqs_action=self.freqs_action,
                freqs_state=self.freqs_state,
                action_register_length=None,
            ).type_as(v)

        return q, k, v, x, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def _apply_video_mixed_attention_output(
        self,
        block,
        residual_x: torch.Tensor,
        mixed_attn_out: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        y = block.self_attn.o(torch.nan_to_num(mixed_attn_out.flatten(2)))
        x = residual_x + y * gate_msa
        if self.model_type == "t2v":
            context_lens = torch.full(
                (x.shape[0],),
                context.shape[1],
                dtype=torch.long,
                device=x.device,
            )
            x = x + block.cross_attn(block.norm3(x), context, context_lens)
        else:
            x = x + block.cross_attn(block.norm3(x), context)
        y = block.ffn(block.norm2(x) * (1 + scale_mlp) + shift_mlp)
        return x + y * gate_mlp

    def _apply_mot_shared_cross_attention_output(
        self,
        block,
        action_block: DreamZeroActionExpertBlock,
        video_residual: torch.Tensor,
        action_residual: torch.Tensor,
        mixed_video: torch.Tensor,
        mixed_action: torch.Tensor,
        video_gate_msa: torch.Tensor,
        video_shift_mlp: torch.Tensor,
        video_scale_mlp: torch.Tensor,
        video_gate_mlp: torch.Tensor,
        action_gate_msa: torch.Tensor,
        action_shift_mlp: torch.Tensor,
        action_scale_mlp: torch.Tensor,
        action_gate_mlp: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_y = block.self_attn.o(torch.nan_to_num(mixed_video.flatten(2)))
        video_x = video_residual + video_y * video_gate_msa

        action_y = torch.nan_to_num(action_block.o(mixed_action.flatten(2)))
        action_x = action_residual + action_y * action_gate_msa

        video_cross_query = block.norm3(video_x)
        action_cross_query = action_block.action_to_video_cross_proj(action_block.norm3(action_x))
        action_cross_query = action_cross_query.to(dtype=video_cross_query.dtype, device=video_cross_query.device)
        joint_query = torch.cat([video_cross_query, action_cross_query], dim=1)

        if self.model_type == "t2v":
            context_lens = torch.full(
                (joint_query.shape[0],),
                context.shape[1],
                dtype=torch.long,
                device=joint_query.device,
            )
            joint_cross = block.cross_attn(joint_query, context, context_lens)
        else:
            joint_cross = block.cross_attn(joint_query, context)

        video_cross, action_cross = joint_cross.split([video_x.shape[1], action_x.shape[1]], dim=1)
        video_x = video_x + video_cross
        action_cross = action_block.video_to_action_cross_proj(action_cross.to(dtype=action_x.dtype))
        action_x = action_x + torch.nan_to_num(action_cross) * action_block.cross_gate.to(
            dtype=action_x.dtype,
            device=action_x.device,
        )

        video_y = block.ffn(block.norm2(video_x) * (1 + video_scale_mlp) + video_shift_mlp)
        video_x = video_x + video_y * video_gate_mlp

        action_y = action_block.norm2(action_x) * (1 + action_scale_mlp) + action_shift_mlp
        action_x = action_x + torch.nan_to_num(action_block.ffn(action_y)) * action_gate_mlp
        return video_x, action_x

    def _build_video_structural_attention_mask(
        self,
        video_seq_len: int,
        clean_seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask = torch.zeros((video_seq_len, video_seq_len), dtype=torch.bool, device=device)

        def fill_first_and_blocks(offset: int, length: int, *, local: bool) -> None:
            first_end = min(self.frame_seqlen, length)
            if first_end > 0:
                mask[offset : offset + first_end, offset : offset + first_end] = True

            block_size = self.frame_seqlen * self.num_frame_per_block
            for block_start in range(self.frame_seqlen, length, block_size):
                block_end = min(block_start + block_size, length)
                if local and self.local_attn_size != -1:
                    kv_start = max(0, block_end - self.local_attn_size * self.frame_seqlen)
                else:
                    kv_start = 0
                mask[
                    offset + block_start : offset + block_end,
                    offset + kv_start : offset + block_end,
                ] = True

        if clean_seq_len > 0:
            if clean_seq_len > video_seq_len:
                raise ValueError(
                    f"clean_seq_len={clean_seq_len} exceeds video_seq_len={video_seq_len}."
                )
            noisy_seq_len = video_seq_len - clean_seq_len
            fill_first_and_blocks(0, clean_seq_len, local=False)

            first_end = min(self.frame_seqlen, noisy_seq_len)
            if first_end > 0:
                noisy_offset = clean_seq_len
                mask[
                    noisy_offset : noisy_offset + first_end,
                    noisy_offset : noisy_offset + first_end,
                ] = True

            block_size = self.frame_seqlen * self.num_frame_per_block
            for block_start in range(self.frame_seqlen, noisy_seq_len, block_size):
                block_end = min(block_start + block_size, noisy_seq_len)
                clean_end = min(block_start, clean_seq_len)
                if clean_end > 0:
                    mask[
                        clean_seq_len + block_start : clean_seq_len + block_end,
                        :clean_end,
                    ] = True
                mask[
                    clean_seq_len + block_start : clean_seq_len + block_end,
                    clean_seq_len + block_start : clean_seq_len + block_end,
                ] = True
        else:
            fill_first_and_blocks(0, video_seq_len, local=True)

        return mask

    def build_mot_mixed_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        clean_seq_len: int = 0,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if device is None:
            device = self.patch_embedding.weight.device
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)
        mask[:video_seq_len, :video_seq_len] = self._build_video_structural_attention_mask(
            video_seq_len=video_seq_len,
            clean_seq_len=clean_seq_len,
            device=device,
        )
        if action_seq_len <= 0:
            return mask

        action_rows = slice(video_seq_len, total_seq_len)
        if self.mot_action_video_attention in _MOT_ACTION_FULL_VIDEO_MODES:
            num_state_per_block = self._infer_action_state_per_block(action_seq_len)
            num_blocks, state_length, _ = self._infer_action_block_layout(
                action_register_length=action_seq_len,
                num_state_per_block=num_state_per_block,
            )
            block_video_len = self.frame_seqlen * self.num_frame_per_block
            video_reads_action = self.mot_action_video_attention == "full_video"

            for block_idx in range(num_blocks):
                state_start = video_seq_len + block_idx * num_state_per_block
                state_end = state_start + num_state_per_block
                action_start = video_seq_len + state_length + block_idx * self.num_action_per_block
                action_end = action_start + self.num_action_per_block

                if num_state_per_block > 0:
                    state_indices = torch.arange(state_start, state_end, device=device)
                    mask[state_indices, state_indices] = True

                if clean_seq_len > 0:
                    noisy_seq_len = video_seq_len - clean_seq_len
                    video_block_start = self.frame_seqlen + block_idx * block_video_len
                    video_block_end = min(video_block_start + block_video_len, noisy_seq_len)
                    clean_end = min(video_block_start, clean_seq_len)
                    video_rows = slice(
                        clean_seq_len + video_block_start,
                        clean_seq_len + video_block_end,
                    )
                    visible_video_slices = []
                    if clean_end > 0:
                        visible_video_slices.append(slice(0, clean_end))
                    if video_block_start < video_block_end:
                        visible_video_slices.append(video_rows)
                else:
                    video_block_start = self.frame_seqlen + block_idx * block_video_len
                    video_block_end = min(video_block_start + block_video_len, video_seq_len)
                    visible_video_end = min(
                        self.frame_seqlen + (block_idx + 1) * block_video_len,
                        video_seq_len,
                    )
                    video_rows = slice(video_block_start, video_block_end)
                    visible_video_slices = [slice(0, visible_video_end)] if visible_video_end > 0 else []

                if video_reads_action and video_rows.start < video_rows.stop:
                    if num_state_per_block > 0:
                        mask[video_rows, state_start:state_end] = True
                    mask[video_rows, action_start:action_end] = True

                action_block_rows = slice(action_start, action_end)
                for video_cols in visible_video_slices:
                    if video_cols.start < video_cols.stop:
                        mask[action_block_rows, video_cols] = True
                if num_state_per_block > 0:
                    mask[action_block_rows, state_start:state_end] = True
                mask[action_block_rows, action_start:action_end] = True
            return mask

        num_state_per_block = self._infer_action_state_per_block(action_seq_len)
        num_blocks, state_length, _ = self._infer_action_block_layout(
            action_register_length=action_seq_len,
            num_state_per_block=num_state_per_block,
        )

        if self.mot_action_video_attention == "first_frame":
            first_frame_tokens = min(self.frame_seqlen, video_seq_len)
        elif self.mot_action_video_attention == "none":
            first_frame_tokens = 0
        else:
            raise ValueError(f"Unsupported mot_action_video_attention={self.mot_action_video_attention!r}")

        for block_idx in range(num_blocks):
            state_start = video_seq_len + block_idx * num_state_per_block
            state_end = state_start + num_state_per_block
            action_start = video_seq_len + state_length + block_idx * self.num_action_per_block
            action_end = action_start + self.num_action_per_block

            if num_state_per_block > 0:
                state_indices = torch.arange(state_start, state_end, device=device)
                mask[state_indices, state_indices] = True

            action_block_rows = slice(action_start, action_end)
            if first_frame_tokens > 0:
                mask[action_block_rows, :first_frame_tokens] = True
            if num_state_per_block > 0:
                mask[action_block_rows, state_start:state_end] = True
            mask[action_block_rows, action_start:action_end] = True

        return mask

    @staticmethod
    def _scaled_dot_product_mixed_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor,
        key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out_dtype = q.dtype
        q_sdpa = q.transpose(1, 2)
        k_sdpa = k.transpose(1, 2)
        v_sdpa = v.transpose(1, 2)
        attn_mask = attention_mask.to(device=q.device, dtype=torch.bool)
        if attn_mask.ndim != 2:
            raise ValueError(f"attention_mask must have shape [Q,K], got {tuple(attn_mask.shape)}")
        if attn_mask.shape != (q.shape[1], k.shape[1]):
            raise ValueError(
                "attention_mask shape must match mixed attention Q/K lengths: "
                f"got {tuple(attn_mask.shape)}, expected {(q.shape[1], k.shape[1])}"
            )
        attn_mask = attn_mask[None, None, :, :]

        if key_mask is not None:
            key_mask = key_mask.to(device=q.device, dtype=torch.bool)
            if key_mask.shape != (q.shape[0], k.shape[1]):
                raise ValueError(
                    "key_mask shape must match batch/key lengths: "
                    f"got {tuple(key_mask.shape)}, expected {(q.shape[0], k.shape[1])}"
                )
            attn_mask = attn_mask & key_mask[:, None, None, :]

        empty_rows = ~attn_mask.any(dim=-1, keepdim=True)
        first_key = torch.zeros_like(attn_mask)
        first_key[..., :1] = True
        attn_mask = attn_mask | (empty_rows & first_key)

        out = F.scaled_dot_product_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            attn_mask=attn_mask,
            dropout_p=0.0,
        )
        out = out.masked_fill(empty_rows, 0)
        return out.transpose(1, 2).contiguous().to(out_dtype)

    def _run_mot_mixed_attention_layer(
        self,
        block,
        action_block: DreamZeroActionExpertBlock,
        video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        e: torch.Tensor,
        action_e: torch.Tensor,
        freqs: torch.Tensor,
        text_context: torch.Tensor,
        is_tf: bool,
        clean_seq_len: int,
        video_key_mask: torch.Tensor | None,
        action_key_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (
            q_video,
            k_video,
            v_video,
            video_residual,
            video_gate_msa,
            video_shift_mlp,
            video_scale_mlp,
            video_gate_mlp,
        ) = self._build_video_mixed_attention_io(
            block=block,
            x=video_tokens,
            e=e,
            freqs=freqs,
            is_tf=is_tf,
        )

        action_register_length = action_tokens.shape[1]
        num_state_per_block = self._infer_action_state_per_block(action_register_length)
        (
            q_action,
            k_action,
            v_action,
            action_residual,
            action_gate_msa,
            action_shift_mlp,
            action_scale_mlp,
            action_gate_mlp,
        ) = action_block.build_mixed_attention_io(
            x=action_tokens,
            e=action_e,
            freqs_action=self.freqs_action,
            freqs_state=self.freqs_state,
            action_register_length=action_register_length,
            num_action_per_block=self.num_action_per_block,
            num_state_per_block=num_state_per_block,
        )

        video_key_mask = self._align_token_mask(video_key_mask, q_video.shape[1], q_video.device)
        action_key_mask = self._align_token_mask(action_key_mask, q_action.shape[1], q_action.device)

        if self.mot_action_video_ki:
            attention_mask = self.build_mot_mixed_attention_mask(
                video_seq_len=q_video.shape[1],
                action_seq_len=q_action.shape[1],
                clean_seq_len=clean_seq_len,
                device=q_video.device,
            )
            mixed_key_mask = self._concat_key_masks(
                [video_key_mask, action_key_mask],
                [q_video.shape[1], q_action.shape[1]],
                batch_size=q_video.shape[0],
                device=q_video.device,
            )
            mixed_video = self._scaled_dot_product_mixed_attention(
                q=q_video,
                k=torch.cat([k_video, k_action], dim=1),
                v=torch.cat([v_video, v_action], dim=1),
                attention_mask=attention_mask[: q_video.shape[1]],
                key_mask=mixed_key_mask,
            )
            detached_video_k, detached_video_v = self._detach_video_kv((k_video, v_video))
            assert detached_video_k is not None and detached_video_v is not None
            mixed_action = self._scaled_dot_product_mixed_attention(
                q=q_action,
                k=torch.cat([detached_video_k, k_action], dim=1),
                v=torch.cat([detached_video_v, v_action], dim=1),
                attention_mask=attention_mask[q_video.shape[1] :],
                key_mask=mixed_key_mask,
            )
        else:
            attention_mask = self.build_mot_mixed_attention_mask(
                video_seq_len=q_video.shape[1],
                action_seq_len=q_action.shape[1],
                clean_seq_len=clean_seq_len,
                device=q_video.device,
            )
            mixed_key_mask = self._concat_key_masks(
                [video_key_mask, action_key_mask],
                [q_video.shape[1], q_action.shape[1]],
                batch_size=q_video.shape[0],
                device=q_video.device,
            )
            mixed = self._scaled_dot_product_mixed_attention(
                q=torch.cat([q_video, q_action], dim=1),
                k=torch.cat([k_video, k_action], dim=1),
                v=torch.cat([v_video, v_action], dim=1),
                attention_mask=attention_mask,
                key_mask=mixed_key_mask,
            )
            mixed_video = mixed[:, : q_video.shape[1]]
            mixed_action = mixed[:, q_video.shape[1] :]

        video_tokens, action_tokens = self._apply_mot_shared_cross_attention_output(
            block=block,
            action_block=action_block,
            video_residual=video_residual,
            action_residual=action_residual,
            mixed_video=mixed_video,
            mixed_action=mixed_action,
            video_gate_msa=video_gate_msa,
            video_shift_mlp=video_shift_mlp,
            video_scale_mlp=video_scale_mlp,
            video_gate_mlp=video_gate_mlp,
            action_gate_msa=action_gate_msa,
            action_shift_mlp=action_shift_mlp,
            action_scale_mlp=action_scale_mlp,
            action_gate_mlp=action_gate_mlp,
            context=text_context,
        )
        return video_tokens, action_tokens

    def _build_cached_mot_mixed_attention_mask(
        self,
        cached_video_seq_len: int,
        current_video_seq_len: int,
        action_seq_len: int,
        device: torch.device,
        current_start_frame: int | None = None,
    ) -> torch.Tensor:
        video_key_len = cached_video_seq_len + current_video_seq_len
        total_query_len = current_video_seq_len + action_seq_len
        total_key_len = video_key_len + action_seq_len
        mask = torch.zeros((total_query_len, total_key_len), dtype=torch.bool, device=device)
        block_video_len = self.frame_seqlen * self.num_frame_per_block
        starts_at_first_frame = current_start_frame == 0 if current_start_frame is not None else cached_video_seq_len == 0

        def fill_cached_video_rows(
            *,
            block_idx: int,
            state_key_start: int | None = None,
            state_key_end: int | None = None,
            action_key_start: int | None = None,
            action_key_end: int | None = None,
        ) -> int:
            if starts_at_first_frame:
                if block_idx == 0:
                    first_end = min(self.frame_seqlen, current_video_seq_len)
                    if first_end > 0:
                        mask[:first_end, :first_end] = True
                video_block_start = min(self.frame_seqlen + block_idx * block_video_len, current_video_seq_len)
                visible_video_end = min(
                    self.frame_seqlen + (block_idx + 1) * block_video_len,
                    video_key_len,
                )
            else:
                video_block_start = min(block_idx * block_video_len, current_video_seq_len)
                visible_video_end = min(
                    cached_video_seq_len + (block_idx + 1) * block_video_len,
                    video_key_len,
                )
            video_block_end = min(video_block_start + block_video_len, current_video_seq_len)
            if video_block_start < video_block_end:
                video_rows = slice(video_block_start, video_block_end)
                if visible_video_end > 0:
                    mask[video_rows, :visible_video_end] = True
                if (
                    self.mot_action_video_attention == "full_video"
                    and state_key_start is not None
                    and state_key_end is not None
                    and state_key_start < state_key_end
                ):
                    mask[video_rows, state_key_start:state_key_end] = True
                if (
                    self.mot_action_video_attention == "full_video"
                    and action_key_start is not None
                    and action_key_end is not None
                ):
                    mask[video_rows, action_key_start:action_key_end] = True
            return visible_video_end

        if action_seq_len <= 0:
            regular_video_tokens = (
                max(current_video_seq_len - self.frame_seqlen, 0)
                if starts_at_first_frame
                else current_video_seq_len
            )
            num_video_blocks = (regular_video_tokens + block_video_len - 1) // block_video_len
            if starts_at_first_frame and current_video_seq_len > 0:
                num_video_blocks = max(num_video_blocks, 1)
            for block_idx in range(num_video_blocks):
                fill_cached_video_rows(block_idx=block_idx)
            return mask

        num_state_per_block = self._infer_action_state_per_block(action_seq_len)
        num_blocks, state_length, _ = self._infer_action_block_layout(
            action_register_length=action_seq_len,
            num_state_per_block=num_state_per_block,
        )
        action_key_offset = video_key_len
        action_query_offset = current_video_seq_len

        for block_idx in range(num_blocks):
            state_start = block_idx * num_state_per_block
            state_end = state_start + num_state_per_block
            action_start = state_length + block_idx * self.num_action_per_block
            action_end = action_start + self.num_action_per_block

            state_key_start = action_key_offset + state_start
            state_key_end = action_key_offset + state_end
            action_key_start = action_key_offset + action_start
            action_key_end = action_key_offset + action_end

            visible_video_end = fill_cached_video_rows(
                block_idx=block_idx,
                state_key_start=state_key_start if num_state_per_block > 0 else None,
                state_key_end=state_key_end if num_state_per_block > 0 else None,
                action_key_start=action_key_start,
                action_key_end=action_key_end,
            )

            if num_state_per_block > 0:
                state_query_start = action_query_offset + state_start
                state_query_end = action_query_offset + state_end
                state_query_indices = torch.arange(state_query_start, state_query_end, device=device)
                state_key_indices = torch.arange(state_key_start, state_key_end, device=device)
                mask[state_query_indices, state_key_indices] = True

            action_query_start = action_query_offset + action_start
            action_query_end = action_query_offset + action_end
            if visible_video_end > 0:
                mask[action_query_start:action_query_end, :visible_video_end] = True
            if num_state_per_block > 0:
                mask[action_query_start:action_query_end, state_key_start:state_key_end] = True
            mask[action_query_start:action_query_end, action_key_start:action_key_end] = True

        return mask

    def _run_mot_mixed_attention_layer_cached(
        self,
        block,
        action_block: DreamZeroActionExpertBlock,
        video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        e: torch.Tensor,
        action_e: torch.Tensor,
        freqs: torch.Tensor,
        text_context: torch.Tensor,
        kv_cache: torch.Tensor,
        current_start_frame: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            q_video,
            k_video,
            v_video,
            video_residual,
            video_gate_msa,
            video_shift_mlp,
            video_scale_mlp,
            video_gate_mlp,
        ) = self._build_video_mixed_attention_io(
            block=block,
            x=video_tokens,
            e=e,
            freqs=freqs,
            is_tf=False,
        )

        action_register_length = action_tokens.shape[1]
        (
            q_action,
            k_action,
            v_action,
            action_residual,
            action_gate_msa,
            action_shift_mlp,
            action_scale_mlp,
            action_gate_mlp,
        ) = action_block.build_mixed_attention_io(
            x=action_tokens,
            e=action_e,
            freqs_action=None,
            freqs_state=None,
            action_register_length=None,
            num_action_per_block=None,
            num_state_per_block=None,
        )
        q_action = self._apply_cached_action_rope(
            q_action,
            action_register_length=action_register_length,
            current_start_frame=current_start_frame,
        )
        k_action = self._apply_cached_action_rope(
            k_action,
            action_register_length=action_register_length,
            current_start_frame=current_start_frame,
        )

        cached_k = kv_cache[0]
        cached_v = kv_cache[1]
        video_k = torch.cat([cached_k, k_video], dim=1)
        video_v = torch.cat([cached_v, v_video], dim=1)
        max_attention_size = int(getattr(block.self_attn, "max_attention_size", video_k.shape[1]))
        if video_k.shape[1] > max_attention_size:
            video_k = video_k[:, -max_attention_size:]
            video_v = video_v[:, -max_attention_size:]
        cached_video_seq_len = max(video_k.shape[1] - k_video.shape[1], 0)
        updated_kv_cache = torch.stack([video_k, video_v], dim=0)

        attention_mask = self._build_cached_mot_mixed_attention_mask(
            cached_video_seq_len=cached_video_seq_len,
            current_video_seq_len=q_video.shape[1],
            action_seq_len=q_action.shape[1],
            device=q_video.device,
            current_start_frame=current_start_frame,
        )
        mixed = self._scaled_dot_product_mixed_attention(
            q=torch.cat([q_video, q_action], dim=1),
            k=torch.cat([video_k, k_action], dim=1),
            v=torch.cat([video_v, v_action], dim=1),
            attention_mask=attention_mask,
        )
        mixed_video = mixed[:, : q_video.shape[1]]
        mixed_action = mixed[:, q_video.shape[1] :]

        video_tokens, action_tokens = self._apply_mot_shared_cross_attention_output(
            block=block,
            action_block=action_block,
            video_residual=video_residual,
            action_residual=action_residual,
            mixed_video=mixed_video,
            mixed_action=mixed_action,
            video_gate_msa=video_gate_msa,
            video_shift_mlp=video_shift_mlp,
            video_scale_mlp=video_scale_mlp,
            video_gate_mlp=video_gate_mlp,
            action_gate_msa=action_gate_msa,
            action_shift_mlp=action_shift_mlp,
            action_scale_mlp=action_scale_mlp,
            action_gate_mlp=action_gate_mlp,
            context=text_context,
        )
        return video_tokens, action_tokens, updated_kv_cache

    def _run_video_expert_block(
        self,
        block,
        x: torch.Tensor,
        e: torch.Tensor,
        freqs: torch.Tensor,
        context: torch.Tensor,
        is_tf: bool,
        key_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._align_wan_modulation(
            block,
            e,
            x.shape[1],
        )
        attn_input = block.norm1(x) * (1 + scale_msa) + shift_msa
        y, video_k, video_v = self._video_self_attention_with_kv(
            self_attn=block.self_attn,
            x=attn_input,
            freqs=freqs,
            is_tf=is_tf,
            key_mask=key_mask,
        )
        y = block.self_attn.o(y)
        x = x + y * gate_msa
        if self.model_type == "t2v":
            context_lens = torch.full(
                (x.shape[0],),
                context.shape[1],
                dtype=torch.long,
                device=x.device,
            )
            x = x + block.cross_attn(block.norm3(x), context, context_lens)
        else:
            x = x + block.cross_attn(block.norm3(x), context)
        y = block.ffn(block.norm2(x) * (1 + scale_mlp) + shift_mlp)
        x = x + y * gate_mlp
        return x, video_k, video_v

    def _select_video_kv_for_action(
        self,
        video_k: torch.Tensor,
        video_v: torch.Tensor,
        seq_len: int,
        clean_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        mode = self.mot_action_video_attention
        if mode == "none":
            return None
        if mode == "first_frame":
            end = min(self.frame_seqlen, video_k.shape[1])
            return video_k[:, :end], video_v[:, :end]
        if mode in _MOT_ACTION_FULL_VIDEO_MODES:
            return video_k, video_v
        raise ValueError(f"Unsupported mot_action_video_attention={self.mot_action_video_attention!r}")

    def _select_video_mask_for_action(
        self,
        video_key_mask: torch.Tensor | None,
        seq_len: int,
        clean_seq_len: int,
    ) -> torch.Tensor | None:
        if video_key_mask is None:
            return None
        mode = self.mot_action_video_attention
        if mode == "none":
            return None
        if mode == "first_frame":
            end = min(self.frame_seqlen, video_key_mask.shape[1])
            return self._drop_full_true_mask(video_key_mask[:, :end])
        if mode in _MOT_ACTION_FULL_VIDEO_MODES:
            return self._drop_full_true_mask(video_key_mask)
        raise ValueError(f"Unsupported mot_action_video_attention={self.mot_action_video_attention!r}")

    def _select_cached_video_kv_for_action(
        self,
        updated_kv_cache: torch.Tensor,
        prefix_kv_cache: torch.Tensor | None = None,
        current_video_token_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        mode = self.mot_action_video_attention
        if mode == "none":
            return None

        video_k = updated_kv_cache[0]
        video_v = updated_kv_cache[1]
        if mode == "first_frame":
            if prefix_kv_cache is not None and prefix_kv_cache.shape[2] > 0:
                video_k = prefix_kv_cache[0]
                video_v = prefix_kv_cache[1]
            end = min(self.frame_seqlen, video_k.shape[1])
            return video_k[:, :end], video_v[:, :end]
        if mode in _MOT_ACTION_FULL_VIDEO_MODES:
            if (
                prefix_kv_cache is not None
                and prefix_kv_cache.shape[2] > 0
                and current_video_token_len is not None
                and video_k.shape[1] <= int(current_video_token_len)
            ):
                # Some refreshed-video paths can hand the action expert only the
                # current generated chunk. Restore the observed prefix explicitly.
                video_k = torch.cat([prefix_kv_cache[0], video_k], dim=1)
                video_v = torch.cat([prefix_kv_cache[1], video_v], dim=1)
                if self.local_attn_size != -1:
                    max_tokens = self.local_attn_size * self.frame_seqlen
                    video_k = video_k[:, -max_tokens:]
                    video_v = video_v[:, -max_tokens:]
            return video_k, video_v
        raise ValueError(f"Unsupported mot_action_video_attention={self.mot_action_video_attention!r}")

    @staticmethod
    def _detach_video_kv(
        video_kv: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if video_kv is None:
            return None
        return video_kv[0].detach(), video_kv[1].detach()

    def _infer_action_state_per_block(self, action_register_length: int) -> int:
        action_state_block = self.num_action_per_block + self.num_state_per_block
        if self.num_state_per_block > 0 and action_register_length % action_state_block == 0:
            return self.num_state_per_block
        if action_register_length % self.num_action_per_block == 0:
            return 0
        raise ValueError(
            "MoT action expert expects registers grouped as action tokens or "
            "state+action tokens: "
            f"action_register_length={action_register_length}, "
            f"num_action_per_block={self.num_action_per_block}, "
            f"num_state_per_block={self.num_state_per_block}."
        )

    def _infer_action_block_layout(
        self,
        action_register_length: int,
        num_state_per_block: int,
    ) -> tuple[int, int, int]:
        per_block = self.num_action_per_block + num_state_per_block
        if per_block <= 0 or action_register_length % per_block != 0:
            raise ValueError(
                "MoT action expert expects registers to form complete causal blocks: "
                f"action_register_length={action_register_length}, "
                f"per_block={per_block}."
            )
        num_blocks = action_register_length // per_block
        state_length = num_blocks * num_state_per_block
        action_length = num_blocks * self.num_action_per_block
        return num_blocks, state_length, action_length

    @staticmethod
    def _slice_optional_key_mask(
        mask: torch.Tensor | None,
        start: int,
        end: int,
    ) -> torch.Tensor | None:
        if mask is None:
            return None
        return mask[:, start:end]

    def _run_action_expert_block_joint_causal(
        self,
        block: DreamZeroActionExpertBlock,
        q_action: torch.Tensor,
        k_action: torch.Tensor,
        v_action: torch.Tensor,
        video_kv: tuple[torch.Tensor, torch.Tensor] | None,
        action_key_mask: torch.Tensor | None,
        video_key_mask: torch.Tensor | None,
        num_state_per_block: int,
        clean_seq_len: int,
        cached_current_start_frame: int | None = None,
    ) -> torch.Tensor:
        action_register_length = q_action.shape[1]
        batch_size = q_action.shape[0]
        device = q_action.device
        num_blocks, state_length, _ = self._infer_action_block_layout(
            action_register_length=action_register_length,
            num_state_per_block=num_state_per_block,
        )
        action_key_mask = self._align_token_mask(action_key_mask, action_register_length, device)

        video_k = None
        video_v = None
        video_length = 0
        if video_kv is not None:
            video_k, video_v = video_kv
            video_length = video_k.shape[1]
            video_key_mask = self._align_token_mask(video_key_mask, video_length, device)

        clean_length = min(max(clean_seq_len, 0), video_length)
        block_video_len = self.frame_seqlen * self.num_frame_per_block
        mixed = torch.empty_like(q_action)

        for block_idx in range(num_blocks):
            state_start = block_idx * num_state_per_block
            state_end = state_start + num_state_per_block
            if num_state_per_block > 0:
                state_mask = self._slice_optional_key_mask(action_key_mask, state_start, state_end)
                mixed[:, state_start:state_end] = block.attn(
                    q_action[:, state_start:state_end],
                    k_action[:, state_start:state_end],
                    v_action[:, state_start:state_end],
                    key_mask=self._drop_full_true_mask(state_mask),
                )

            action_start = state_length + block_idx * self.num_action_per_block
            action_end = action_start + self.num_action_per_block
            k_parts = []
            v_parts = []
            key_masks = []

            if video_k is not None and video_length > 0:
                if clean_length > 0:
                    clean_end = min(self.frame_seqlen + block_idx * block_video_len, clean_length)
                    if clean_end > 0:
                        k_parts.append(video_k[:, :clean_end])
                        v_parts.append(video_v[:, :clean_end])
                        key_masks.append(self._slice_optional_key_mask(video_key_mask, 0, clean_end))

                    noisy_start = clean_length + self.frame_seqlen + block_idx * block_video_len
                    noisy_end = min(noisy_start + block_video_len, video_length)
                    if noisy_start < noisy_end:
                        k_parts.append(video_k[:, noisy_start:noisy_end])
                        v_parts.append(video_v[:, noisy_start:noisy_end])
                        key_masks.append(self._slice_optional_key_mask(video_key_mask, noisy_start, noisy_end))
                else:
                    if cached_current_start_frame is None:
                        visible_video_end = min(
                            self.frame_seqlen + (block_idx + 1) * block_video_len,
                            video_length,
                        )
                    else:
                        visible_video_end = min(
                            (cached_current_start_frame + (block_idx + 1) * self.num_frame_per_block)
                            * self.frame_seqlen,
                            video_length,
                        )
                    if visible_video_end > 0:
                        k_parts.append(video_k[:, :visible_video_end])
                        v_parts.append(video_v[:, :visible_video_end])
                        key_masks.append(self._slice_optional_key_mask(video_key_mask, 0, visible_video_end))

            if num_state_per_block > 0:
                k_parts.append(k_action[:, state_start:state_end])
                v_parts.append(v_action[:, state_start:state_end])
                key_masks.append(self._slice_optional_key_mask(action_key_mask, state_start, state_end))

            k_parts.append(k_action[:, action_start:action_end])
            v_parts.append(v_action[:, action_start:action_end])
            key_masks.append(self._slice_optional_key_mask(action_key_mask, action_start, action_end))

            key_mask = self._concat_key_masks(
                key_masks,
                [part.shape[1] for part in k_parts],
                batch_size=batch_size,
                device=device,
            )
            mixed[:, action_start:action_end] = block.attn(
                q_action[:, action_start:action_end],
                torch.cat(k_parts, dim=1),
                torch.cat(v_parts, dim=1),
                key_mask=key_mask,
            )

        return mixed.flatten(2)

    def _apply_cached_action_rope(
        self,
        x: torch.Tensor,
        action_register_length: int,
        current_start_frame: int,
    ) -> torch.Tensor:
        num_state_per_block = self._infer_action_state_per_block(action_register_length)
        per_block = self.num_action_per_block + num_state_per_block
        if action_register_length % per_block != 0:
            raise ValueError(
                "MoT cached inference expects action registers to be grouped by "
                f"state+action block size ({per_block}), got "
                f"action_register_length={action_register_length}."
            )

        action_state_index = max((current_start_frame - 1) // self.num_frame_per_block, 0)
        action_offset = action_state_index * self.num_action_per_block
        state_offset = action_state_index * num_state_per_block
        empty_video_freqs = self.freqs_action.new_empty((0, 1, self.freqs_action.shape[-1]))
        ordered_freqs = _ordered_state_action_freqs(
            freqs_action=self.freqs_action,
            freqs_state=self.freqs_state,
            action_register_length=action_register_length,
            num_action_per_block=self.num_action_per_block,
            num_state_per_block=num_state_per_block,
            action_offset=action_offset,
            state_offset=state_offset,
        )
        return _apply_rope_with_ordered_freqs(x, empty_video_freqs, ordered_freqs).type_as(x)

    def _run_video_expert_block_cached(
        self,
        block,
        x: torch.Tensor,
        e: torch.Tensor,
        freqs: torch.Tensor,
        context: torch.Tensor,
        kv_cache: torch.Tensor,
        current_start_frame: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._align_wan_modulation(
            block,
            e,
            x.shape[1],
        )
        attn_input = block.norm1(x) * (1 + scale_msa) + shift_msa
        y, updated_kv_cache = block.self_attn(
            x=attn_input,
            freqs=freqs,
            freqs_action=self.freqs_action,
            freqs_state=self.freqs_state,
            action_register_length=None,
            kv_cache=kv_cache,
            current_start_frame=current_start_frame,
            is_tf=False,
        )
        assert updated_kv_cache is not None
        x = x + y * gate_msa
        if self.model_type == "t2v":
            context_lens = torch.full(
                (x.shape[0],),
                context.shape[1],
                dtype=torch.long,
                device=x.device,
            )
            x = x + block.cross_attn(block.norm3(x), context, context_lens)
        else:
            x = x + block.cross_attn(block.norm3(x), context)
        y = block.ffn(block.norm2(x) * (1 + scale_mlp) + shift_mlp)
        x = x + y * gate_mlp
        return x, updated_kv_cache

    def _run_action_expert_block(
        self,
        block: DreamZeroActionExpertBlock,
        tokens: torch.Tensor,
        e: torch.Tensor,
        context: torch.Tensor | None,
        video_kv: tuple[torch.Tensor, torch.Tensor] | None,
        action_key_mask: torch.Tensor | None = None,
        video_key_mask: torch.Tensor | None = None,
        clean_seq_len: int = 0,
    ) -> torch.Tensor:
        action_register_length = tokens.shape[1]
        num_state_per_block = self._infer_action_state_per_block(action_register_length)
        (
            q_action,
            k_action,
            v_action,
            residual_tokens,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = block.build_mixed_attention_io(
            x=tokens,
            e=e,
            freqs_action=self.freqs_action,
            freqs_state=self.freqs_state,
            action_register_length=action_register_length,
            num_action_per_block=self.num_action_per_block,
            num_state_per_block=num_state_per_block,
        )

        mixed = self._run_action_expert_block_joint_causal(
            block=block,
            q_action=q_action,
            k_action=k_action,
            v_action=v_action,
            video_kv=video_kv,
            action_key_mask=action_key_mask,
            video_key_mask=video_key_mask,
            num_state_per_block=num_state_per_block,
            clean_seq_len=clean_seq_len,
            cached_current_start_frame=None,
        )
        return block.apply_mixed_attention_output(
            residual_x=residual_tokens,
            mixed_attn_out=mixed,
            gate_msa=gate_msa,
            shift_mlp=shift_mlp,
            scale_mlp=scale_mlp,
            gate_mlp=gate_mlp,
            context=context,
        )

    def _run_action_expert_block_cached(
        self,
        block: DreamZeroActionExpertBlock,
        tokens: torch.Tensor,
        e: torch.Tensor,
        context: torch.Tensor | None,
        video_kv: tuple[torch.Tensor, torch.Tensor] | None,
        current_start_frame: int,
    ) -> torch.Tensor:
        action_register_length = tokens.shape[1]
        (
            q_action,
            k_action,
            v_action,
            residual_tokens,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = block.build_mixed_attention_io(
            x=tokens,
            e=e,
            freqs_action=None,
            freqs_state=None,
            action_register_length=None,
            num_action_per_block=None,
            num_state_per_block=None,
        )
        q_action = self._apply_cached_action_rope(
            q_action,
            action_register_length=action_register_length,
            current_start_frame=current_start_frame,
        )
        k_action = self._apply_cached_action_rope(
            k_action,
            action_register_length=action_register_length,
            current_start_frame=current_start_frame,
        )

        num_state_per_block = self._infer_action_state_per_block(action_register_length)
        mixed = self._run_action_expert_block_joint_causal(
            block=block,
            q_action=q_action,
            k_action=k_action,
            v_action=v_action,
            video_kv=video_kv,
            action_key_mask=None,
            video_key_mask=None,
            num_state_per_block=num_state_per_block,
            clean_seq_len=0,
            cached_current_start_frame=current_start_frame,
        )
        return block.apply_mixed_attention_output(
            residual_x=residual_tokens,
            mixed_attn_out=mixed,
            gate_msa=gate_msa,
            shift_mlp=shift_mlp,
            scale_mlp=scale_mlp,
            gate_mlp=gate_mlp,
            context=context,
        )

    def _forward_inference(
        self,
        x,
        timestep,
        context,
        seq_len,
        kv_cache: list[torch.Tensor],
        crossattn_cache: list[torch.Tensor],
        current_start_frame: int,
        y=None,
        clip_feature=None,
        action=None,
        timestep_action=None,
        state=None,
        embodiment_id=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        if self.model_type == "i2v":
            assert clip_feature is not None and y is not None
        assert context.shape[1] == self.text_len

        if y is not None and self.concat_first_frame_latent:
            x = torch.cat([x, y.to(dtype=x.dtype)], dim=1)

        x = self.patch_embedding(x)
        grid_size = torch.tensor(x.shape[2:], dtype=torch.long)
        freqs = self._create_freqs(
            grid_size=grid_size,
            start_frame=current_start_frame,
        )

        x = x.flatten(start_dim=2).transpose(1, 2)
        assert x.shape[1] == seq_len
        batch_size = x.shape[0]
        video_frames = timestep.shape[1]

        if video_frames <= seq_len:
            repeat = (seq_len + video_frames - 1) // video_frames
            timestep_video = timestep.repeat_interleave(repeat, dim=1)[:, :seq_len]
        else:
            indices = torch.linspace(
                0,
                video_frames - 1,
                seq_len,
                device=timestep.device,
                dtype=torch.long,
            )
            timestep_video = timestep[:, indices]

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep_video.flatten()).type_as(x)
        )
        e = e.unflatten(dim=0, sizes=timestep_video.shape)
        e0 = self.time_projection(e).unflatten(dim=2, sizes=(6, self.dim))

        if action is not None:
            assert state is not None
            embodiment_id = torch.zeros(batch_size, device=x.device, dtype=torch.long)
        elif embodiment_id is None and state is not None:
            embodiment_id = torch.zeros(batch_size, device=x.device, dtype=torch.long)
        text_image_context = self._build_text_image_context(context, clip_feature)
        text_context = self._build_video_context(text_image_context, None)

        action_tokens = None
        action_e = None
        action_start = None
        action_length = None
        if action is not None:
            assert timestep_action is not None
            assert state is not None
            action_tokens, action_e, action_start, action_length = self.action_expert.build_action_inputs(
                action=action,
                timestep_action=timestep_action,
                state=state,
                embodiment_id=embodiment_id,
            )

        updated_kv_caches: list[torch.Tensor] = []
        for layer_idx, block in enumerate(self.blocks):
            if action_tokens is not None and self.mot_video_reads_action:
                assert action_e is not None
                x, action_tokens, updated_kv_cache = self._run_mot_mixed_attention_layer_cached(
                    block=block,
                    action_block=self.action_expert.blocks[layer_idx],
                    video_tokens=x,
                    action_tokens=action_tokens,
                    e=e0,
                    action_e=action_e,
                    freqs=freqs,
                    text_context=text_context,
                    kv_cache=kv_cache[layer_idx],
                    current_start_frame=current_start_frame,
                )
                updated_kv_caches.append(updated_kv_cache)
                continue

            x, updated_kv_cache = self._run_video_expert_block_cached(
                block=block,
                x=x,
                e=e0,
                freqs=freqs,
                context=text_context,
                kv_cache=kv_cache[layer_idx],
                current_start_frame=current_start_frame,
            )
            updated_kv_caches.append(updated_kv_cache)

            if action_tokens is None:
                continue

            assert action_e is not None
            action_tokens = self._run_action_expert_block_cached(
                block=self.action_expert.blocks[layer_idx],
                tokens=action_tokens,
                e=action_e,
                context=None,
                video_kv=self._select_cached_video_kv_for_action(
                    updated_kv_cache,
                    prefix_kv_cache=kv_cache[layer_idx],
                    current_video_token_len=seq_len,
                ),
                current_start_frame=current_start_frame,
            )

        x_video = self.head(x[:, :seq_len], e[:, :seq_len].unsqueeze(2))
        video_noise_pred = self.unpatchify(x_video, grid_size)

        action_noise_pred = None
        if action_tokens is not None:
            assert action_start is not None
            assert action_length is not None
            assert embodiment_id is not None
            action_noise_pred = self.action_expert.decode_action(
                tokens=action_tokens,
                action_start=action_start,
                action_length=action_length,
                embodiment_id=embodiment_id,
            )

        return video_noise_pred, action_noise_pred, updated_kv_caches

    def _forward_action_from_video_kv_cache(
        self,
        action,
        timestep_action,
        state,
        embodiment_id,
        kv_cache: list[torch.Tensor],
        current_start_frame: int,
        require_full_video: bool = False,
        prefix_kv_cache: list[torch.Tensor] | None = None,
        current_video_token_len: int | None = None,
    ) -> torch.Tensor:
        """Run only the MoT action expert using a supplied per-layer video K/V cache."""
        if self.mot_video_reads_action:
            raise RuntimeError(
                "Action-only video-cache inference is incompatible with bidirectional MoT attention; "
                "use joint denoise inference instead."
            )
        if require_full_video and self.mot_action_video_attention not in _MOT_ACTION_FULL_VIDEO_MODES:
            raise ValueError(
                "MoT refreshed-video action path requires "
                "mot_action_video_attention=full_video or full_video_unidirectional."
            )
        batch_size = action.shape[0]
        # Match the existing cached inference path, which uses the default
        # action embodiment category when action registers are present.
        embodiment_id = torch.zeros(batch_size, device=action.device, dtype=torch.long)

        action_tokens, action_e, action_start, action_length = self.action_expert.build_action_inputs(
            action=action,
            timestep_action=timestep_action,
            state=state,
            embodiment_id=embodiment_id,
        )

        for layer_idx, block in enumerate(self.action_expert.blocks):
            layer_prefix_kv = None
            if prefix_kv_cache is not None:
                layer_prefix_kv = prefix_kv_cache[layer_idx]
            layer_video_kv = self._select_cached_video_kv_for_action(
                kv_cache[layer_idx],
                prefix_kv_cache=layer_prefix_kv,
                current_video_token_len=current_video_token_len,
            )

            if torch.is_grad_enabled() and self.action_expert.use_gradient_checkpointing:
                if layer_video_kv is None:

                    def run_cached_action_layer(layer_tokens, layer_e, _block=block):
                        return self._run_action_expert_block_cached(
                            block=_block,
                            tokens=layer_tokens,
                            e=layer_e,
                            context=None,
                            video_kv=None,
                            current_start_frame=current_start_frame,
                        )

                    action_tokens = torch.utils.checkpoint.checkpoint(
                        run_cached_action_layer,
                        action_tokens,
                        action_e,
                        use_reentrant=False,
                    )
                else:
                    layer_video_k, layer_video_v = layer_video_kv

                    def run_cached_action_layer(
                        layer_tokens,
                        layer_e,
                        video_k,
                        video_v,
                        _block=block,
                    ):
                        return self._run_action_expert_block_cached(
                            block=_block,
                            tokens=layer_tokens,
                            e=layer_e,
                            context=None,
                            video_kv=(video_k, video_v),
                            current_start_frame=current_start_frame,
                        )

                    action_tokens = torch.utils.checkpoint.checkpoint(
                        run_cached_action_layer,
                        action_tokens,
                        action_e,
                        layer_video_k,
                        layer_video_v,
                        use_reentrant=False,
                    )
            else:
                action_tokens = self._run_action_expert_block_cached(
                    block=block,
                    tokens=action_tokens,
                    e=action_e,
                    context=None,
                    video_kv=layer_video_kv,
                    current_start_frame=current_start_frame,
                )

        return self.action_expert.decode_action(
            tokens=action_tokens,
            action_start=action_start,
            action_length=action_length,
            embodiment_id=embodiment_id,
        )

    def forward_action_from_cached_video(
        self,
        action,
        timestep_action,
        state,
        embodiment_id,
        kv_cache: list[torch.Tensor],
        current_start_frame: int,
    ) -> torch.Tensor:
        """Run only the MoT action expert using observed/clean video K/V caches."""
        return self._forward_action_from_video_kv_cache(
            action=action,
            timestep_action=timestep_action,
            state=state,
            embodiment_id=embodiment_id,
            kv_cache=kv_cache,
            current_start_frame=current_start_frame,
        )

    def forward_action_from_refreshed_video_kv(
        self,
        action,
        timestep_action,
        state,
        embodiment_id,
        video_kv_cache: list[torch.Tensor],
        current_start_frame: int,
        prefix_kv_cache: list[torch.Tensor] | None = None,
        current_video_token_len: int | None = None,
    ) -> torch.Tensor:
        """Run action expert from the latest full-video denoise refresh K/V."""
        return self._forward_action_from_video_kv_cache(
            action=action,
            timestep_action=timestep_action,
            state=state,
            embodiment_id=embodiment_id,
            kv_cache=video_kv_cache,
            current_start_frame=current_start_frame,
            require_full_video=True,
            prefix_kv_cache=prefix_kv_cache,
            current_video_token_len=current_video_token_len,
        )

    def _forward_train(
        self,
        x,
        timestep,
        timestep_action,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        y=None,
        clip_feature=None,
        action=None,
        state=None,
        embodiment_id=None,
        video_frame_mask=None,
        action_token_mask=None,
        state_token_mask=None,
    ):
        if self.model_type == "i2v":
            assert clip_feature is not None and y is not None

        if y is not None and self.concat_first_frame_latent:
            x = torch.cat([x, y.to(dtype=x.dtype)], dim=1)

        x = self.patch_embedding(x)
        grid_size = torch.tensor(x.shape[2:], dtype=torch.long)
        freqs = self._create_freqs(grid_size=grid_size, start_frame=0)

        x = x.flatten(start_dim=2).transpose(1, 2)
        assert x.shape[1] == seq_len
        batch_size = x.shape[0]
        video_frames = timestep.shape[1]
        video_token_mask = self._build_video_token_mask(
            video_frame_mask=video_frame_mask,
            video_frames=video_frames,
            seq_len=seq_len,
            device=x.device,
        )
        video_attention_mask = video_token_mask

        timestep_video = timestep.unsqueeze(-1).expand(
            batch_size, video_frames, seq_len // video_frames
        ).reshape(batch_size, -1)
        timestep_original = timestep_video.clone()

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep_video.flatten()).type_as(x)
        )
        e = e.unflatten(dim=0, sizes=timestep_video.shape)
        e0 = self.time_projection(e).unflatten(dim=2, sizes=(6, self.dim))

        assert context.shape[1] == self.text_len
        if action is not None:
            assert state is not None
            embodiment_id = torch.zeros(batch_size, device=x.device, dtype=torch.long)
        elif embodiment_id is None and state is not None:
            embodiment_id = torch.zeros(batch_size, device=x.device, dtype=torch.long)
        text_image_context = self._build_text_image_context(context, clip_feature)
        text_context = self._build_video_context(text_image_context, None)

        clean_seq_len = 0
        if clean_x is not None:
            if y is not None and self.concat_first_frame_latent:
                clean_x = torch.cat([clean_x, y.to(dtype=clean_x.dtype)], dim=1)
            clean_x = self.patch_embedding(clean_x)
            clean_x = clean_x.flatten(start_dim=2).transpose(1, 2)
            assert clean_x.shape[1] == seq_len

            x = torch.cat([clean_x, x], dim=1)
            clean_seq_len = clean_x.shape[1]
            if video_token_mask is not None:
                video_attention_mask = torch.cat([video_token_mask, video_token_mask], dim=1)

            if aug_t is None:
                aug_t = torch.zeros_like(timestep_original)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x)
            )
            e_clean = e_clean.unflatten(dim=0, sizes=timestep_original.shape)
            e0_clean = self.time_projection(e_clean).unflatten(dim=2, sizes=(6, self.dim))
            e0 = torch.cat([e0_clean, e0], dim=1)

        action_tokens = None
        action_e = None
        action_start = None
        action_length = None
        action_register_mask = None
        if action is not None:
            assert timestep_action is not None
            assert state is not None
            action_tokens, action_e, action_start, action_length = self.action_expert.build_action_inputs(
                action=action,
                timestep_action=timestep_action,
                state=state,
                embodiment_id=embodiment_id,
            )
            state_mask = self._align_token_mask(state_token_mask, action_start, x.device)
            action_mask = self._align_token_mask(action_token_mask, action_length, x.device)
            action_register_mask = self._concat_key_masks(
                [state_mask, action_mask],
                [action_start, action_length],
                batch_size=batch_size,
                device=x.device,
            )

        is_tf = clean_x is not None
        for layer_idx, block in enumerate(self.blocks):
            if action_tokens is None:
                def run_video(video_tokens, _block=block):
                    video_tokens, _, _ = self._run_video_expert_block(
                        block=_block,
                        x=video_tokens,
                        e=e0,
                        freqs=freqs,
                        context=text_context,
                        is_tf=is_tf,
                        key_mask=video_attention_mask,
                    )
                    return video_tokens

                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(run_video, x, use_reentrant=False)
                else:
                    x = run_video(x)
                continue

            assert action_e is not None
            action_block = self.action_expert.blocks[layer_idx]

            if not self.mot_video_reads_action:
                def run_separate_layer(video_tokens, action_tokens_in, _block=block, _action_block=action_block):
                    video_tokens, video_k, video_v = self._run_video_expert_block(
                        block=_block,
                        x=video_tokens,
                        e=e0,
                        freqs=freqs,
                        context=text_context,
                        is_tf=is_tf,
                        key_mask=video_attention_mask,
                    )
                    video_kv = self._select_video_kv_for_action(
                        video_k=video_k,
                        video_v=video_v,
                        seq_len=seq_len,
                        clean_seq_len=clean_seq_len,
                    )
                    video_kv_mask = self._select_video_mask_for_action(
                        video_attention_mask,
                        seq_len=seq_len,
                        clean_seq_len=clean_seq_len,
                    )
                    if self.mot_action_video_ki:
                        video_kv = self._detach_video_kv(video_kv)
                    action_tokens_out = self._run_action_expert_block(
                        block=_action_block,
                        tokens=action_tokens_in,
                        e=action_e,
                        context=None,
                        video_kv=video_kv,
                        action_key_mask=action_register_mask,
                        video_key_mask=video_kv_mask,
                        clean_seq_len=clean_seq_len,
                    )
                    return video_tokens, action_tokens_out

                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    x, action_tokens = torch.utils.checkpoint.checkpoint(
                        run_separate_layer,
                        x,
                        action_tokens,
                        use_reentrant=False,
                    )
                else:
                    x, action_tokens = run_separate_layer(x, action_tokens)
                continue

            def run_mot_layer(video_tokens, action_tokens_in, _block=block, _action_block=action_block):
                return self._run_mot_mixed_attention_layer(
                    block=_block,
                    action_block=_action_block,
                    video_tokens=video_tokens,
                    action_tokens=action_tokens_in,
                    e=e0,
                    action_e=action_e,
                    freqs=freqs,
                    text_context=text_context,
                    is_tf=is_tf,
                    clean_seq_len=clean_seq_len,
                    video_key_mask=video_attention_mask,
                    action_key_mask=action_register_mask,
                )

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x, action_tokens = torch.utils.checkpoint.checkpoint(
                    run_mot_layer,
                    x,
                    action_tokens,
                    use_reentrant=False,
                )
            else:
                x, action_tokens = run_mot_layer(x, action_tokens)

        if clean_x is not None:
            x = x[:, clean_seq_len:]

        x_video = x[:, :seq_len]
        e_video = e[:, :seq_len]
        x_video = self.head(x_video, e_video.unsqueeze(2))
        video_noise_pred = self.unpatchify(x_video, grid_size)

        action_noise_pred = None
        if action_tokens is not None:
            assert action_start is not None
            assert action_length is not None
            assert embodiment_id is not None
            action_noise_pred = self.action_expert.decode_action(
                tokens=action_tokens,
                action_start=action_start,
                action_length=action_length,
                embodiment_id=embodiment_id,
            )

        return video_noise_pred, action_noise_pred
