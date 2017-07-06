import gym
import numpy       as np
import pandas      as pd
import tensorflow  as tf
import tensorflow.contrib.slim as slim

from functools import *

__all__ = [
    "gym",
    "np",
    "pd",
    "tf",
    "slim",

    "partial",
    "partialmethod",
    "reduce",

    "_p",
    "mul",
    "div",
    "add",
    "sub",
    "compose",
    "_c",
    "eligibility_trace",
    "oneof",
    "lmap",
    "npmap",
    "imap",
    "_0",
    "_1",
    "_2",
    "_3",
    "_4",
    "choose_action",
    "one_hot_encode",
    "zipWith",
    "space_sizes",
    "flatten_tensor",
    "Writer",
    "EpisodeWriter"
    ]

_p = partial

mul = lambda b: (lambda a: a * b)
div = lambda b: (lambda a: a / b)
add = lambda b: (lambda a: a + b)
sub = lambda b: (lambda a: a - b)

def compose(*functions):
    """ compose """
    def inner(arg):
        for f in reversed(functions):
            arg = f(arg)
        return arg
    return inner

_c = compose

def oneof(fn1, fn2):
    tmp = None

    try:
      tmp = fn1()
    except:
      tmp = fn2()

    return tmp

def lmap(fn, ls):
    return list(map(fn, ls))

def npmap(fn, ls):
    return np.array(lmap(fn, ls))

def imap(fn, ls):
    return lmap(lambda xs: fn(xs[0], xs[1]), enumerate(ls))

_0 = lambda t: t[0]
_1 = lambda t: t[1]
_2 = lambda t: t[2]
_3 = lambda t: t[3]
_4 = lambda t: t[4]

def choose_action(dist, probs):
    return np.argmax(dist == np.random.choice(probs, p=probs))

def one_hot_encode(i, total):
  return np.identity(total)[i:i+1]

def zipWith(fn, _as, _bs):
    return list(map(lambda gs: fn(gs[0], gs[1]), zip(_as, _bs)))


""" Shared reinforcement learning functions"""

def eligibility_trace(gamma, rs):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(rs)
    running_add = 0
    size = oneof(lambda: rs.size, lambda: len(rs))

    for t in reversed(range(0, size)):
        running_add = running_add * gamma + rs[t]
        discounted_r[t] = running_add
    return discounted_r

def space_sizes(env):
    a_size = env.action_space.n
    try:
      return a_size, env.observation_space.n
    except:
      return a_size, env.observation_space.shape[0]

def flatten_tensor(tns):
    return tf.reshape(tns, [-1])


class Writer:
    def __init__(self):
        self.log = []

    def tell(self, *args):
        self.log.append(list(args))

    def listen(self):
        return self.log


class EpisodeWriter(Writer):
    def __init__(self, dtype=None):
        super(EpisodeWriter, self).__init__()
        assert type(dtype) == type, "dtype must be a type or None"
        self.dtype = dtype

    def tell(self, observed_state, reward, action, isdone):
        super(EpisodeWriter, self).tell(observed_state, reward, action, isdone)

    def listen(self):
        return np.array(super(EpisodeWriter, self).listen(), dtype=self.dtype)


