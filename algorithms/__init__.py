from abc import ABC

class Checkpointable(ABC):

    def get_replay_buffer(self):
        return getattr(self, 'replay_buffer', None)

    def get_reservoir_buffer(self):
        return getattr(self, '_reservoir_buffer', None)

    def load_replay_buffer(self, buffer):
        if hasattr(self, 'replay_buffer'):
            self.replay_buffer = buffer

    def load_reservoir_buffer(self, buffer):
        if hasattr(self, '_reservoir_buffer'):
            self._reservoir_buffer = buffer

