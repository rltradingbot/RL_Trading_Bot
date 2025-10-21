"""
Torch-based experience replay buffer (CPU-only storage, tensor-only API).

- Stores transitions as (state, action, reward, next_state, done) CPU tensors
- Any incoming GPU tensors are detached and moved to CPU on push (no graph leaks)
- Sampling returns a TransitionBatch with (B, ...) CPU tensors
- TransitionBatch supports .to()/.pin_memory() for one-shot copying/move

Usage
-----
Basic usage with single-step pushes and sampling:

```python
import torch
from replay_buffer import ReplayBuffer

buffer = ReplayBuffer(capacity=100_000, seed=42, ready_after=1000)

# Example shapes (adjust to your environment):
# state: (obs_dim,), action: (...), reward: (1,), next_state: (obs_dim,), done: (1,) bool
state = torch.randn(8)
action = torch.tensor([1])
reward = torch.tensor([0.5])
next_state = torch.randn(8)
done = torch.tensor([False])

buffer.push(state, action, reward, next_state, done)

if buffer.is_ready():
    batch = buffer.sample(batch_size=64, replace=False)  # returns TransitionBatch
    # Move to GPU in one shot if needed
    batch = batch.to('cuda')
    states, actions, rewards, next_states, dones = batch.as_tuple()
```

Batch push when you already have a batch of transitions:

```python
B = 32
states = torch.randn(B, 8)
actions = torch.randint(0, 3, (B, 1))
rewards = torch.randn(B, 1)
next_states = torch.randn(B, 8)
dones = torch.zeros(B, 1, dtype=torch.bool)

buffer.push_batch(states, actions, rewards, next_states, dones)
```

Notes
-----
- Inputs can be on CPU or GPU; they will be detached and moved to CPU internally.
- `done` is stored as `torch.bool`.
- Sampling with `replace=False` requires `batch_size <= len(buffer)`.
"""

from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple
from collections import deque
import random
import torch


def _to_cpu_detached(x: torch.Tensor, name: str) -> torch.Tensor:
    """Ensure tensor input; detach from grad graph; move to CPU."""
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x)}")
    # detach + move to CPU; clone not needed unless caller mutates; keep minimal copy
    t = x.detach()
    return t.cpu() if t.device.type != "cpu" else t


@dataclass(frozen=True, slots=True)
class Transition:
    """Single transition tuple stored in the buffer.

    Fields are CPU tensors. Shapes are user-defined, but common conventions are:
    - state: (...)
    - action: (...)
    - reward: (1,)
    - next_state: (...)
    - done: (1,) with dtype bool
    """
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    done: torch.Tensor  # dtype=bool, CPU


@dataclass(frozen=True, slots=True)
class TransitionBatch:
    """Batched tensors with convenience methods for one-shot device transfer."""
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor

    def to(self, *args, **kwargs) -> "TransitionBatch":
        """Return a new batch with all fields moved/copied by torch.Tensor.to."""
        return TransitionBatch(
            self.states.to(*args, **kwargs),
            self.actions.to(*args, **kwargs),
            self.rewards.to(*args, **kwargs),
            self.next_states.to(*args, **kwargs),
            self.dones.to(*args, **kwargs),
        )

    def pin_memory(self) -> "TransitionBatch":
        """Pin all tensors (useful before async device transfer)."""
        return TransitionBatch(
            self.states.pin_memory(),
            self.actions.pin_memory(),
            self.rewards.pin_memory(),
            self.next_states.pin_memory(),
            self.dones.pin_memory(),
        )

    def as_tuple(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.states, self.actions, self.rewards, self.next_states, self.dones


class ReplayBuffer:
    """A simple FIFO replay buffer returning batched torch tensors (CPU-only storage).

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to retain. Acts as a circular buffer.
    seed : Optional[int]
        Seed for internal RNG used during sampling.
    ready_after : int
        Minimum number of stored transitions before `is_ready()` returns True.

    Characteristics
    ---------------
    - Push accepts CPU or GPU tensors; they are detached and stored on CPU.
    - Sampling returns a `TransitionBatch` of stacked CPU tensors.
    - Use `TransitionBatch.to(device)` to move the batch where needed.
    """

    def __init__(self, capacity: int, seed: Optional[int] = None, ready_after: int = 1) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if ready_after <= 0:
            raise ValueError("ready_after must be >= 1")
        self._capacity: int = int(capacity)
        self._storage: Deque[Transition] = deque(maxlen=self._capacity)
        self._rng: random.Random = random.Random(seed)
        self._ready_after: int = int(ready_after)

    # replay_buffer.push: input state, action, reward, next_state, done (all tensors)
    def push(self, state: torch.Tensor, action: torch.Tensor,
             reward: torch.Tensor, next_state: torch.Tensor, done: torch.Tensor) -> None:
        """Insert a single transition.

        All inputs must be tensors. If they are on GPU, they are detached and moved to CPU.
        The `done` tensor is converted to dtype bool.
        """
        s  = _to_cpu_detached(state, "state")
        a  = _to_cpu_detached(action, "action")
        r  = _to_cpu_detached(reward, "reward")
        ns = _to_cpu_detached(next_state, "next_state")
        d  = _to_cpu_detached(done, "done").to(torch.bool)
        self._storage.append(Transition(s, a, r, ns, d))

    # Optional batch push: insert (B, ...) tensors at once
    def push_batch(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor,
                   next_states: torch.Tensor, dones: torch.Tensor) -> None:
        """Insert a batch of transitions with leading dimension B.

        All inputs must have the same leading dimension. Tensors are detached and
        moved to CPU as needed. `dones` are converted to dtype bool.
        """
        S  = _to_cpu_detached(states, "states")
        A  = _to_cpu_detached(actions, "actions")
        R  = _to_cpu_detached(rewards, "rewards")
        NS = _to_cpu_detached(next_states, "next_states")
        D  = _to_cpu_detached(dones, "dones").to(torch.bool)

        if not (S.size(0) == A.size(0) == R.size(0) == NS.size(0) == D.size(0)):
            raise ValueError("All batched inputs must have the same leading dimension")
        B = S.size(0)
        # Unroll and append in one pass (Python loop remains; conversion already done)
        for i in range(B):
            self._storage.append(Transition(S[i], A[i], R[i], NS[i], D[i]))

    # replay_buffer.sample: input batch_size, output batched CPU tensors (B, ...)
    def sample(self, batch_size: int, replace: bool = False) -> TransitionBatch:
        """Randomly sample a batch of transitions.

        Parameters
        ----------
        batch_size : int
            Number of elements in the returned batch.
        replace : bool
            If True, sample with replacement; otherwise without replacement.

        Returns
        -------
        TransitionBatch
            Batched CPU tensors of shape (batch_size, ...).
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        num_items = len(self._storage)
        if num_items == 0:
            raise ValueError("cannot sample from an empty buffer")

        if replace:
            indices: List[int] = [self._rng.randrange(num_items) for _ in range(batch_size)]
        else:
            if batch_size > num_items:
                raise ValueError(
                    f"batch_size ({batch_size}) cannot be greater than current size ({num_items}) when replace=False"
                )
            indices = self._rng.sample(range(num_items), k=batch_size)

        # Gather directly from the deque to avoid O(N) list materialization
        storage = self._storage
        states      = torch.stack([storage[i].state      for i in indices], dim=0).contiguous()
        actions     = torch.stack([storage[i].action     for i in indices], dim=0).contiguous()
        rewards     = torch.stack([storage[i].reward     for i in indices], dim=0).contiguous()
        next_states = torch.stack([storage[i].next_state for i in indices], dim=0).contiguous()
        dones       = torch.stack([storage[i].done       for i in indices], dim=0).contiguous()

        return TransitionBatch(states, actions, rewards, next_states, dones)

    def clear(self) -> None:
        """Remove all stored transitions."""
        self._storage.clear()

    def size(self) -> int:
        """Current number of stored transitions."""
        return len(self._storage)

    def is_empty(self) -> bool:
        """True if the buffer has no stored transitions."""
        return len(self._storage) == 0

    def is_full(self) -> bool:
        """True if the buffer has reached or exceeded its capacity."""
        return len(self._storage) >= self._capacity

    def is_ready(self) -> bool:
        """True if the number of transitions >= `ready_after`."""
        return len(self._storage) >= self._ready_after

    def __len__(self) -> int:
        return len(self._storage)

    def __repr__(self) -> str:
        return f"ReplayBuffer(size={len(self._storage)}, capacity={self._capacity})"


__all__ = ["Transition", "TransitionBatch", "ReplayBuffer"]
