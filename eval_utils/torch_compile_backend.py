from __future__ import annotations

import functools
import os
from typing import Any

import torch


def configure_torch_compile_backend(default_backend: str | None = None) -> str | None:
    """Patch ``torch.compile`` for entrypoints that need a safer backend.

    This is intentionally scoped to launcher / server entrypoints so we can
    switch the runtime backend without mutating the shared training
    environment. Users can still override the choice via
    ``TORCH_COMPILE_BACKEND``.
    """

    requested_backend = os.getenv("TORCH_COMPILE_BACKEND")
    backend = requested_backend if requested_backend is not None else default_backend
    if not backend or getattr(torch.compile, "_dreamzero_backend_patched", False):
        return backend

    # Propagate the entrypoint-selected backend to downstream imports in this
    # process so module-level compile decisions can stay backend-aware without
    # mutating the shared virtualenv or requiring shell-level env setup.
    if requested_backend is None:
        os.environ["TORCH_COMPILE_BACKEND"] = backend

    real_compile = torch.compile

    @functools.wraps(real_compile)
    def wrapped_compile(*args: Any, **kwargs: Any):
        kwargs.setdefault("backend", backend)
        # Non-Inductor backends such as cudagraphs do not use Inductor's mode
        # presets; dropping the argument avoids backend-specific validation
        # differences across PyTorch versions.
        if kwargs.get("backend") != "inductor":
            kwargs.pop("mode", None)
        return real_compile(*args, **kwargs)

    wrapped_compile._dreamzero_backend_patched = True  # type: ignore[attr-defined]
    torch.compile = wrapped_compile
    return backend
