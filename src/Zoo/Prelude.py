import numpy as np
from functools import *

__all__ = [
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
    "imap",
    "_0",
    "_1",
    "_2",
    "choose_action",
    "one_hot_encode",
    "zipWith"
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

def eligibility_trace(gamma, rs):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(rs)
    running_add = 0
    size = oneof(lambda: rs.size, lambda: len(rs))

    for t in reversed(range(0, size)):
        running_add = running_add * gamma + rs[t]
        discounted_r[t] = running_add
    return discounted_r

def oneof(fn1, fn2):
    tmp = None

    try:
      tmp = fn1()
    except:
      tmp = fn2()

    return tmp

def lmap(fn, ls):
    return list(map(fn, ls))

def imap(fn, ls):
    return lmap(lambda xs: fn(xs[0], xs[1]), enumerate(ls))

_0 = lambda t: t[0]
_1 = lambda t: t[1]
_2 = lambda t: t[2]

def choose_action(dist, probs):
    return np.argmax(dist == np.random.choice(probs, p=probs))

def one_hot_encode(i, total):
  return np.identity(total)[i:i+1]

def zipWith(fn, _as, _bs):
    return list(map(lambda gs: fn(gs[0], gs[1]), zip(_as, _bs)))

