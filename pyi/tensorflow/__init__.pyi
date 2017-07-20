from typing import *

class ConfigProto(Any): ...

class Variable(Any):
  def __init__(a:Any, dtype:Any=None)->None:...

class Tensor(Any): ...
class GraphKeys(Any): ...

class nn(Any):
  def relu (a:Any)->Tensor: ...

truncated_normal_initializer: Any

float32: Any

random_uniform_initializer: Any
named_scope: Any

placeholder: Any

get_collection: Any

name_scope: Any

reshape: Any

zeros: Any

matmul: Any

app: Any

def multiply(a:Union[Any, int], b:Union[Any, int])->Tensor: ...
def tanh(a:Any)->Any: ...


