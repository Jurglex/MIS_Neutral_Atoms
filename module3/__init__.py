from __future__ import annotations

from .interfaces import Learner, Orchestrator, ScheduleNetwork, TrainingConfig

__all__ = [
    "Learner",
    "Orchestrator",
    "ReinforceLearner",
    "ScheduleNetwork",
    "TrainingConfig",
    "TrainingOrchestrator",
]


def __getattr__(name: str):
    """Lazy-load concrete implementations to keep base imports light."""
    if name == "ReinforceLearner":
        from .learner import ReinforceLearner
        return ReinforceLearner
    if name == "TrainingOrchestrator":
        from .orchestrator import TrainingOrchestrator
        return TrainingOrchestrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
