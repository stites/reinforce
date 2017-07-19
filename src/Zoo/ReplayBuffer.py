from Zoo.Prelude import *
from collections import deque

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
    def __init__(self, buffer_size:int=50000):
        self.buffer:Deque[Any] = deque()
        self.buffer_size:int = buffer_size
        self.count:int = 0

    def clear(self)->None:
        self.buffer.clear()
        self.count = 0

    def add_step(self, s, a, r, s1, done)->None:
        self._resize(1)
        self.append(np.reshape(np.array([s,a,r,s1,done]),[1,5]))

    def append(self, experience)->None:
        self._resize(len(experience))
        self.buffer.append(experience)

    def add_episode(self, episode)->None:
        self._resize(1)
        self.buffer.append(episode)

    def _resize(self, item_size:int)->None:
        """ okay, get rid of num_extras and just use count+while """
        if self.count + item_size >= self.buffer_size:
            num_extras = lqueue + item_size - self.buffer_size
            while num_extras > 0:
                self.buffer.popleft()
                num_extras -= 1
                self.count -= 1

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


