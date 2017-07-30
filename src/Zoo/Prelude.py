import gym
import numpy       as np
import pandas      as pd
import tensorflow  as tf
import tensorflow.contrib.slim as slim

import random
import os
import csv
import itertools
import multiprocessing

from functools import *
from typing    import *

__all__ = [
    "gym",
    "np",
    "pd",
    "tf",
    "slim",
    "random",
    "os",
    "csv",
    "itertools",

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
    "lzip",

    "head",
    "tail",
    "last",
    "init",

    "_0",
    "_1",
    "_2",
    "_3",
    "_4",
    "choose_action",
    "one_hot_encode",
    "zipWith",
    "curry",
    "truncate",

    "set_global_random_seed",
    "epsilon_greedy",
    "AnnealingEpsilon",

    "space_sizes",
    "flatten_tensor",
    "Writer",
    "EpisodeWriter",
    "ReportWriter",

    "cpus",
    "sess_config"
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

def lzip(*args):
    return list(zip(*args))

def head(ls):
    return ls[0]

def tail(ls):
    return ls[1:]

def last(ls):
    return ls[-1]

def init(ls):
    return ls[:-1]

def identity1(x):
    return x

def identityArgs(*args):
    return args

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

def curry(fn):
    return (lambda a: fn(a[0], a[1]))

def lambda2(fn):
    return (lambda a: fn(a[0], a[1]))

def lambda3(fn):
    return (lambda a: fn(a[0], a[1], a[2]))

def lambda4(fn):
    return (lambda a: fn(a[0], a[1], a[2], a[4]))


def truncate(f):
    #assert type(f) == float, "truncate is for float only"
    return float('%.6f'%(f))

""" Shared reinforcement learning functions"""

def set_global_random_seed(seed, env=None):
    if not (env is None):
        env.seed(seed)

    np.random.seed(seed)
    tf.set_random_seed(seed)


def eligibility_trace(gamma, rs):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(rs)
    running_add = 0
    size = oneof(lambda: rs.size, lambda: len(rs))

    for t in reversed(range(0, size)):
        running_add = running_add * gamma + rs[t]
        discounted_r[t] = running_add
    return discounted_r.astype('float64')

def space_sizes(env):
    a_size = env.action_space.n
    try:
      return a_size, env.observation_space.n
    except:
      return a_size, env.observation_space.shape[0]


def flatten_tensor(tns):
    return tf.reshape(tns, [-1])


def epsilon_greedy(eps, action_size, action_chooser, other_conditions=lambda: False):
    choose_random = np.random.rand(1) < eps or other_conditions()
    return np.random.randint(0, action_size) if choose_random \
      else action_chooser()


class AnnealingEpsilon:
    def __init__(self, step_size, start_step=0, eps_range=(1,0)):
        self._is_end     = False
        self.eps_range   = eps_range
        self.start,self.end   = eps_range

        self.step_size   = float(step_size)
        self.start_step  = float(step_size)
        self.start_decay = lambda curr_step: start_step < curr_step

    def linear_decay(self, step):
        start, end = self.eps_range
        return ((start - end) / self.step_size) * (step - self.start_step)

    def __call__(self, step):
        start, end = self.eps_range
        if self._is_end:
            return end
        elif not self.start_decay(step):
            return start
        else:
            calculated  = start - self.linear_decay(step)
            self._is_end = calculated < end
            return end if self._is_end else calculated


class Writer(object):
    def __init__(self):
        self.log = []

    def tell(self, *args):
        self.log.append(list(args))

    def listen(self):
        return self.log

    def length(self):
        return len(self.log)

class EpisodeWriter(Writer):
    def __init__(self, dtype=None):
        super(EpisodeWriter, self).__init__()
        assert type(dtype) == type or dtype == None, "dtype must be a type or None"
        self.dtype = dtype

    def tell(self, observed_state, reward, action, isdone):
        super(EpisodeWriter, self).tell(observed_state, float(reward), int(action), int(isdone))
        assert type(observed_state) == np.ndarray, "observed state must be recorded as a numpy array"

    def tellAll(self, observed_state, reward, action, isdone, advantages, state_value):
        super(EpisodeWriter, self).tell(observed_state, reward, action, isdone, advantages, state_value)
        assert type(observed_state) == np.ndarray, "observed state must be recorded as a numpy array"
        assert type(reward) == float, "reward must be recorded as a float"
        assert type(action) == int, "action must be recorded as an int"
        assert type(isdone) == int, "done must be recorded as an int"

    def listen(self):
        return np.array(super(EpisodeWriter, self).listen(), dtype=self.dtype)

    def states(self):
        return np.vstack(self.listen()[:,0])

    def actions(self):
        return self.listen()[:,1][np.newaxis].T

    def rewards(self):
        return self.listen()[:,2][np.newaxis].T

    def dones(self):
        return self.listen()[:,3][np.newaxis].T

class ReportWriter(Writer):
    def __init__(self):
        super(ReportWriter, self).__init__()

    def tell(self, episodewriter):
        assert type(episodewriter) == EpsisodeWriter, "must be an episode writer"
        rs = episodewriter.rewards()
        super(ReportWriter, self).tell(episodewriter, rs.sum(), len(rs))

        # episode #
        # episode length
        # episode total rwd
        # episode details:
        # -> action, reward, (forall a Action a=>. advantage(a)), Value, State

cpus = multiprocessing.cpu_count()
sess_config = tf.ConfigProto(intra_op_parallelism_threads=cpus, inter_op_parallelism_threads=cpus)

