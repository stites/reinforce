from Zoo.Prelude import *
from collections import deque

import warnings
import functools

__all__ = ["ReplayBuffer"]

def depreciated(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        msg = "Call to deprecated function {}.".format(func.__name__)
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)
        return func(*args, **kwargs)

    return new_func

class ReplayBuffer:
    def __init__(self, buffer_size:int=50000)->None:
        self.buffer:Deque[Any] = deque()
        self.buffer_size:int = buffer_size
        self.count:int = 0

    def clear(self)->None:
        self.buffer.clear()
        self.count = 0

    def append(self, experience)->None:
        """ this is the main function which increments the counter """
        self._resize(len(experience))
        self.buffer.append(experience)
        self.count += len(experience)

    def add_step(self, s:Any, a:int, r:float, s1:Any, done:bool)->None:
        """ alias to append """
        self._resize(1)
        self.append(np.reshape(np.array([s,a,r,s1,done]),[1,5]))

    def add_episode(self, episode)->None:
        """ alias to append """
        self._resize(1)
        self.buffer.append(episode)

    def _resize(self, item_size:int)->None:
        """ okay, get rid of num_extras and just use count+while """
        if self.count + item_size >= self.buffer_size:
            num_extras = self.count + item_size - self.buffer_size
            while num_extras > 0:
                self.buffer.popleft()
                num_extras -= 1
                self.count -= 1

    def sample_sequence(self, size, trace_length):
        """ sample traces """
        sampled_episodes = random.sample(self.buffer, size)
        sampledTraces = []

        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode) + 1 - trace_length)
            sampledTraces.append( episode[point:point+trace_length] )

        return np.reshape(np.array(sampledTraces),[size*trace_length, 5])

    def sample_batch(self, requested_size):
        """ sample batches """
        batch_size = self.count if self.count < requested_size else requested_size
        return np.reshape(np.array(random.sample(self.buffer, batch_size)), [batch_size, 5])

    @depreciated
    def sample(self, requested_size):
        """ alias to sample_batches """
        return self.sample_batch(requested_size)

    def sample_batch_split(self, requested_size:int)->Tuple[Any, Any, Any, Any, Any]:
        """ alias to sample_batches but splits output """
        batch      = self.sample_batch(requested_size)
        batch_size = batch.shape[0]

        def reshape(split):
            is_stacked   = type(split[0]) is np.ndarray
            series_shape = [batch_size, 1]
            return np.vstack(split) if is_stacked else np.reshape(split, series_shape)

        s_batch  = reshape(batch[:,0])
        a_batch  = reshape(batch[:,1])
        r_batch  = reshape(batch[:,2])
        s1_batch = reshape(batch[:,3])
        d_batch  = reshape(batch[:,4])


        return s_batch, a_batch, r_batch, s1_batch, d_batch


