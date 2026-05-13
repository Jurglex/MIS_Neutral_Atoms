from __future__ import annotations

from .interfaces import BackendResult, DeviceMetadata, Positions, QuantumBackend

__all__ = [
    "BackendResult",
    "BraketBackend",
    "DeviceMetadata",
    "Positions",
    "QuantumBackend",
]


def __getattr__(name: str):
    """Lazy-load BraketBackend so amazon-braket-sdk is not required at import."""
    if name == "BraketBackend":
        from .braket_backend import BraketBackend

        return BraketBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
