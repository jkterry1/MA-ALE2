import pickle
import random
import torch
import numpy as np
from typing import Dict
import time
from lz4.frame import compress, decompress

from all.memory import ReplayBuffer, NStepReplayBuffer, ExperienceReplayBuffer, PrioritizedReplayBuffer
from all.core import State


class CompressedPrioritizedReplayBuffer(PrioritizedReplayBuffer):
    def __init__(self, buffer_size, compress=True, **kwargs):
        super().__init__(buffer_size, **kwargs)
        self.compress = compress

    def store(self, state, action, next_state):
        if state is None or state.done:
            return
        idx = self.pos
        state = state.to(self.store_device)
        next_state = next_state.to(self.store_device)
        if self.compress:
            state = compress(pickle.dumps(state))
            next_state = compress(pickle.dumps(next_state))
        self._add((state, action, next_state))
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _reshape(self, minibatch, weights):
        states = State.array([pickle.loads(decompress(sample[0])) if self.compress else sample[0]
                              for sample in minibatch]).to(self.device)
        if torch.is_tensor(minibatch[0][1]):
            actions = torch.stack([sample[1] for sample in minibatch]).to(self.device)
        else:
            actions = torch.tensor([sample[1] for sample in minibatch], device=self.device)
        next_states = State.array([pickle.loads(decompress(sample[2])) if self.compress else sample[2]
                              for sample in minibatch]).to(self.device)
        return (states, actions, next_states.reward, next_states, weights)


class ParallelNStepBuffer(ReplayBuffer):

    def __init__(self, steps, discount_factor, buffer, n_envs):
        # Since NStepReplay just wraps buffer, keep n_envs of these.
        # Now we will add each transition to its index in self._nstep_buffers
        # and each will handle storing in the (shared) underlying buffer.
        self.n_envs = n_envs
        self._nstep_buffers = [NStepReplayBuffer(steps, discount_factor, buffer) for _ in range(n_envs)]
        self.buffer = buffer
        assert all(rb.buffer is self.buffer for rb in self._nstep_buffers)


    def store(self, states, actions, next_states):
        if states is None:
            return
        assert len(actions) == self.n_envs

        not_done_idxs = (~states.done).nonzero().flatten().tolist()
        actions = actions.to(self.buffer.store_device)
        for i in not_done_idxs:
            self._nstep_buffers[i].store(states[i], actions[i], next_states[i])

    def sample(self, *args, **kwargs):
        return self.buffer.sample(*args, **kwargs)

    def update_priorities(self, *args, **kwargs):
        return self.buffer.update_priorities(*args, **kwargs)

    def __len__(self):
        return len(self.buffer)


class ReservoirBuffer(ExperienceReplayBuffer):
    ''' Allows uniform sampling over a stream of data.

    This class supports the storage of arbitrary elements, such as observation
    tensors, integer actions, etc.

    See https://en.wikipedia.org/wiki/Reservoir_sampling for more details.
    '''

    def __init__(self, size, device='cpu', store_device=None, compress=False):
        super().__init__(size, device=device, store_device=store_device)
        self._add_calls = 0
        self.compress = compress

    def _add(self, sample):
        """Potentially adds `element` to the reservoir buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            idx = np.random.randint(0, self._add_calls + 1)
            if idx < self.capacity:
                self.buffer[idx] = sample
        self._add_calls += 1

    def store(self, state, action, next_state):
        if state is not None and not state.done:
            state = state.to(self.store_device)
            action = action.to(self.store_device)
            if self.compress:
                state = compress(pickle.dumps(state))
            self._add((state, action, None))

    def sample(self, num_samples: int):
        """Returns `num_samples` uniformly sampled from the buffer."""
        if len(self.buffer) < num_samples:
            raise ValueError(f"{num_samples} elements could not be sampled from size {len(self.buffer)}")
        minibatch = random.sample(self.buffer, num_samples)
        return self._reshape(minibatch, torch.ones(num_samples, device=self.device))

    def _reshape(self, minibatch, weights):
        states = State.array([pickle.loads(decompress(sample[0])) if self.compress else sample[0]
                              for sample in minibatch]).to(self.device)
        if torch.is_tensor(minibatch[0][1]):
            actions = torch.stack([sample[1] for sample in minibatch]).to(self.device)
        else:
            actions = torch.tensor([sample[1] for sample in minibatch], device=self.device)
        return (states, actions, None, None, weights)


class ParallelReservoirBuffer(ReservoirBuffer):
    """ResevoirBuffer compatible with ParallelAgents"""

    def store(self, states, actions, next_states):
        if states is None:
            return

        not_done_idxs = (~states.done).nonzero().flatten().tolist()
        for i in not_done_idxs:
            state = states[i].to(self.store_device)
            action = actions[i].to(self.store_device)
            if self.compress:
                state = compress(pickle.dumps(state))
            self._add((state, action, None))
