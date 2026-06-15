"""Replay buffer for off-policy reuse of expensive simulator rollouts.

Stores tuples of ``(graph_idx, data, sampled_params, old_logprob, old_value,
reward, baseline_reward, omega, delta)`` with a FIFO eviction policy.  When
mixed into a PPO update, the importance-sampling ratio
``π_new(a|s) / π_old(a|s)`` is computed from stored ``old_logprob``; PPO's
clipping then bounds the effective off-policy distance.

Sample-efficiency rationale
---------------------------
Each (graph, schedule) → reward call costs a full Braket simulator run.
A buffer of size B let us reuse each rollout for up to roughly B / batch
gradient steps before staleness dominates.  With PPO clipping at ε=0.2 the
typical effective horizon is 5–10 steps of staleness.
"""
from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np
import torch


class ReplayBuffer:
    """FIFO buffer holding rollout tuples for off-policy PPO updates.

    Parameters
    ----------
    capacity : int
        Maximum number of stored entries.  ``capacity <= 0`` disables the
        buffer (``add`` becomes a no-op and ``sample`` returns ``None``).
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = max(int(capacity), 0)
        self._buf: deque = deque(maxlen=self.capacity) if self.capacity > 0 else deque()

    def __len__(self) -> int:
        return len(self._buf)

    def add_rollouts(self, rollouts: dict) -> None:
        """Append rollouts from a ``collect_rollouts`` dict (one entry per
        row).  ``rollouts['data_list']`` is sliced by ``graph_idx`` so
        each stored entry carries the PyG data needed to recompute log-probs.
        """
        if self.capacity <= 0:
            return
        N = rollouts["sampled_params"].shape[0]
        graph_idx = rollouts["graph_idx"].cpu().numpy()
        data_list = rollouts["data_list"]
        for i in range(N):
            entry = {
                "data": data_list[int(graph_idx[i])],
                "sampled_params": rollouts["sampled_params"][i].cpu(),
                "old_logprob": rollouts["old_logprob"][i].cpu(),
                "old_value": rollouts["old_value"][i].cpu(),
                "reward": rollouts["reward"][i].cpu(),
                "baseline_reward": rollouts["baseline_reward"][i].cpu(),
                "omega": rollouts["omega"][i].cpu(),
                "delta": rollouts["delta"][i].cpu(),
            }
            self._buf.append(entry)

    def sample(self, k: int) -> dict | None:
        """Draw ``k`` random entries (with replacement) and return a
        rollouts dict matching the schema of ``collect_rollouts``.
        Returns ``None`` if the buffer is empty.
        """
        if len(self._buf) == 0 or k <= 0:
            return None
        rng = np.random.default_rng()
        idx = rng.integers(0, len(self._buf), size=k)
        entries = [self._buf[int(j)] for j in idx]
        return _stack_entries(entries)

    def clear(self) -> None:
        self._buf.clear()


def _stack_entries(entries: Iterable[dict]) -> dict:
    entries = list(entries)
    out = {
        "sampled_params": torch.stack([e["sampled_params"] for e in entries]),
        "old_logprob": torch.stack([e["old_logprob"] for e in entries]),
        "old_value": torch.stack([e["old_value"] for e in entries]),
        "reward": torch.stack([e["reward"] for e in entries]).float(),
        "baseline_reward": torch.stack([e["baseline_reward"] for e in entries]).float(),
        "omega": torch.stack([e["omega"] for e in entries]),
        "delta": torch.stack([e["delta"] for e in entries]),
        "data_list": [e["data"] for e in entries],
        "graph_idx": torch.arange(len(entries), dtype=torch.long),
    }
    return out
