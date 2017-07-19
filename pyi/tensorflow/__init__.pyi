from typing import *

class ConfigProto(Any): ...
class Variable(Any): ...
class Tensor(Any): ...

class nn(Any):
  def relu (a:Any)->Tensor: ...

truncated_normal_initializer: Any

float32: Any

random_uniform_initializer: Any

placeholder: Any

reshape: Any

zeros: Any

matmul: Any

app: Any

def multiply(a:Union[Any, int], b:Union[Any, int])->Tensor: ...
def tanh(a:Any)->Any: ...


