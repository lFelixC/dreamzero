from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class RTCSessionState:
    last_action_chunk_abs: np.ndarray | None = None
    consumed_steps: int = 0
    chunk_start_step_idx: int | None = None
    request_count: int = 0


def extract_optional_int(value, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        value = value.item() if value.size == 1 else value.reshape(-1)[0]
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def session_key(session_id: str | None) -> str:
    return str(session_id) if session_id is not None else "__default__"


def trim_action_chunk_for_delay(
    action_chunk: np.ndarray,
    inference_delay_steps: int,
) -> tuple[np.ndarray, int]:
    if action_chunk.shape[0] <= 1 or inference_delay_steps <= 0:
        return action_chunk, 0

    trim_steps = min(inference_delay_steps, action_chunk.shape[0] - 1)
    return action_chunk[trim_steps:], trim_steps


def compute_prev_chunk_left_over(
    session_state: RTCSessionState,
    rtc_step_idx: int | None,
) -> np.ndarray | None:
    if session_state.last_action_chunk_abs is None:
        return None
    if rtc_step_idx is None or session_state.chunk_start_step_idx is None:
        return None

    consumed_steps = max(int(rtc_step_idx) - int(session_state.chunk_start_step_idx), 0)
    consumed_steps = min(consumed_steps, session_state.last_action_chunk_abs.shape[0])
    session_state.consumed_steps = consumed_steps
    if consumed_steps >= session_state.last_action_chunk_abs.shape[0]:
        return None
    return session_state.last_action_chunk_abs[consumed_steps:]
