import contextlib
import os
from collections.abc import Iterator

import torch


_TRUE_VALUES = {"1", "true", "yes", "on"}


def nvtx_enabled() -> bool:
    return os.environ.get("DREAMZERO_ENABLE_NVTX", "").strip().lower() in _TRUE_VALUES


@contextlib.contextmanager
def nvtx_range(name: str) -> Iterator[None]:
    if not nvtx_enabled():
        yield
        return

    pushed = False
    try:
        torch.cuda.nvtx.range_push(str(name))
        pushed = True
    except Exception:
        yield
        return

    try:
        yield
    finally:
        if pushed:
            torch.cuda.nvtx.range_pop()
