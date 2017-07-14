from Zoo.Prelude import *

class ReplayBuffer:
    def __init__(self, buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add_step(self, s, a, r, s1, done):
        self._resize(1)
        self.append(np.reshape(np.array([s,a,r,s1,done]),[1,5]))

    def append(self, experience):
        self._resize(len(experience))
        self.buffer.extend(experience)

    def add_episode(self, episode):
        self._resize(1)
        self.buffer.append(episode)

    def _resize(self, item_size):
        lqueue = len(self.buffer)
        if lqueue + item_size >= self.buffer_size:
            self.buffer[0:(lqueue + item_size - self.buffer_size)] = []

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])

    def sample_sequence(self, size, trace_length):
        sampled_episodes = random.sample(self.buffer, size)
        sampledTraces = []

        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode) + 1 - trace_length)
            sampledTraces.append( episode[point:point+trace_length] )

        return np.reshape(np.array(sampledTraces),[size*trace_length, 5])


