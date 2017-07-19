from Zoo.Prelude      import *
from Zoo.ReplayBuffer import *

# weight initializers
def truncated_normal_initializer(in_size:int, out_size:int):
    return tf.truncated_normal_initializer(
         [in_size, out_size],
         mean=0.0,
         stddev=0.02,
         dtype=tf.float32)


def uniform_initializer(in_size:int, out_size:int):
    return tf.random_uniform_initializer([in_size, out_size], minval=-0.003, maxval=0.003)

# layer definitions
class Layer(object):
    """layer of a neural network"""
    def __init__(self,
            inputs_:Union[Any, int],
            out_size:int,
            weight_initialization:Callable[[int, int], Any],
            activation:Callable[[Any], Any]
        )->None:
        super(Layer, self).__init__()
        batch_size:int = None  # for posterity
        in_size:int    = None

        if isinstance(inputs_, tf.Tensor):
            in_size = inputs_.get_shape().as_list()[1]  # assume this is a 2d tensor
            inputs  = inputs_
        else:
            in_size = inputs_
            inputs  = tf.placeholder(in_size)

        inputs_flat = tf.reshape(inputs, [-1, batch_size])
        weights     = tf.Variable(weight_initialization(in_size, out_size))
        bias        = tf.Variable(tf.zeros([out_size]))
        outputs     = activation(tf.matmul(inputs_flat, weights) + bias)

        self.inputs, self.bias, self.outputs, self.weights = inputs, bias, outputs, weights


class TwoInputLayer(object):
    """ critic layer 2 pulls in the actions as an input """
    def __init__(self,
            input1:Any,
            input2:Any,
            out_size:int,
            weight_initialization:Callable[[int, int], Any],
            activation:Callable[[Any], Any]
        )->None:
        super(TwoInputLayer, self).__init__()
        """ not going to try to modify this to taking an input size at all """

        in_size = input1.get_shape().as_list()[-1]
        assert in_size == input2.get_shape().as_list()[-1], "input sizes do not match"

        weights1 = tf.Variable(weight_initialization(in_size, out_size))
        weights2 = tf.Variable(weight_initialization(in_size, out_size))
        bias     = tf.Variable(tf.zeros([out_size]))
        outputs  = activation(tf.matmul(input1, weights1) + tf.matmul(input2, weights2) + bias)

        self.input1  = input1
        self.input2  = input2
        self.outputs = outputs


class ActorNetwork(object):
    """ build out a deterministic policy network """
    def __init__(self, state_space:int, action_space:int, action_boundary:int)->None:
        super(ActorNetwork, self).__init__()
        self.inputs, self.outputs = self.build(state_space, action_space)
        self.outputs_scaled       = tf.multiply(self.outputs, action_boundary)

    def build(self, state_space:int, action_space:int):
        layer1 = Layer(   state_space,          400, truncated_normal_initializer, tf.nn.relu)
        layer2 = Layer(layer1.outputs,          300, truncated_normal_initializer, tf.nn.relu)
        layer3 = Layer(layer2.outputs, action_space,          uniform_initializer,    tf.tanh)
        return layer1.inputs, layer3.outputs


class CriticNetwork(object):
    """ build out a compatible function approximator """
    def __init__(self, state_space, action_space):
        super(CriticNetwork, self).__init__()
        self.states_input, self.actions_input, self.outputs = self.build(state_space, action_space)

    def build(self, state_space:int, action_space:int):
        states  = tf.placeholder([None,  state_space])
        actions = tf.placeholder([None, action_space])
        layer1 = Layer(states, 400, truncated_normal_initializer, tf.nn.relu)
        layer2 = TwoInputLayer(actions, layer1.outputs, 300, truncated_normal_initializer, tf.nn.relu)
        layer3 = Layer(layer2.outputs, 1, uniform_initializer, identity1)

        return states, actions, layer3.outputs


