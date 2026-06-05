"""Minimal policy interface used by DreamZero websocket eval utilities."""

from __future__ import annotations

import abc
from typing import Any


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: dict[str, Any]) -> Any:
        """Infer actions from observations."""

    def reset(self, reset_info: dict[str, Any] | None = None) -> None:
        """Reset the policy to its initial state."""
        pass
