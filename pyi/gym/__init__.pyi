from typing import *

error:Any
spaces:Any

class Env(object):

  observation_space:Any

  action_space:Any

  def reset(self)->Any: ...

  def step(self, a:int)->Any: ...


def make(a:str)->Env: ...


