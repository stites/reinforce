from Zoo.Prelude import *
import warnings
import functools

def deprecated(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        msg = "Call to deprecated function {}.".format(func.__name__)
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)
        return func(*args, **kwargs)

    return new_func

class ReplayBuffer:
    def __init__(self, buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size

    @property
    def size(self):
        return len(self.buffer)

    def clear(self):
        self.buffer = []

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

    def sample_sequence(self, size, trace_length):
        sampled_episodes = random.sample(self.buffer, size)
        sampledTraces = []

        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode) + 1 - trace_length)
            sampledTraces.append( episode[point:point+trace_length] )

        return np.reshape(np.array(sampledTraces),[size*trace_length, 5])

    @depreciated
    def sample(self, requested_size):
        return self.sample_batch(requested_size)

    def sample_batch(self, requested_size):
        batch_size = self.size if self.size < requested_size else requested_size
        return np.reshape(np.array(random.sample(self.buffer, batch_size)), [batch_size, 5])

    def sample_batch_split(self, requested_size):
        batch    = self.sample_batch(requested_size)

        s_batch  = batch[:,0]
        a_batch  = batch[:,1]
        r_batch  = batch[:,2]
        s1_batch = batch[:,3]
        d_batch  = batch[:,4]

        return s_batch, a_batch, r_batch, s1_batch, d_batch


