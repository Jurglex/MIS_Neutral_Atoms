from __future__ import annotations

from .interfaces import Learner, Orchestrator, ScheduleNetwork, TrainingConfig

__all__ = [
    "Learner",
    "Orchestrator",
    "ReinforceLearner",
    "ReplayBuffer",
    "ScheduleNetwork",
    "TrainingConfig",
    "TrainingOrchestrator",
    "behavioral_clone_critic",
    "behavioral_clone_policy",
    "graph_conditioning_index",
    "ppo_step",
    "reinforce_step",
    "schedule_deviation_probe",
]


def __getattr__(name: str):
    """Lazy-load concrete implementations to keep base imports light."""
    if name == "ReinforceLearner":
        from .learner import ReinforceLearner
        return ReinforceLearner
    if name == "TrainingOrchestrator":
        from .orchestrator import TrainingOrchestrator
        return TrainingOrchestrator
    if name == "ReplayBuffer":
        from .replay import ReplayBuffer
        return ReplayBuffer
    if name == "reinforce_step":
        from .reinforce import reinforce_step
        return reinforce_step
    if name == "ppo_step":
        from .ppo import ppo_step
        return ppo_step
    if name in ("behavioral_clone_policy", "behavioral_clone_critic"):
        from .pretrain import behavioral_clone_policy, behavioral_clone_critic
        return {
            "behavioral_clone_policy": behavioral_clone_policy,
            "behavioral_clone_critic": behavioral_clone_critic,
        }[name]
    if name in ("schedule_deviation_probe", "graph_conditioning_index"):
        from .diagnostics import (
            schedule_deviation_probe, graph_conditioning_index,
        )
        return {
            "schedule_deviation_probe": schedule_deviation_probe,
            "graph_conditioning_index": graph_conditioning_index,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
