from Zoo.Prelude   import *
from Zoo.Gridworld import Gridworld

env = Gridworld(partial=False, sizeX=5)
# convolution_layer = tf.contrib.layers.convolution2d(
#         inputs,       # input layer
# 
#         num_outputs,  # how many filters we would like to apply to the previous layer
# 
#         kernel_size,  # how large a window we would like to slide over the previous layer
# 
#         stride,       # how many pixels we want to skip as we move the window across the layer
# 
#         padding)      # whether we want our window to slide over just the bottom layer
#                       # ("VALID") or add padding around it ("SAME") in order to ensure
#                       # that the convolutional layer has the same dimensions as the 
#                       # previous layer.
"""
Improvements to DQNs since its first release
============================================

### Experience Replay

By storing an agent's experiences and randomly drawing on batches of them for
training, we wind up training on a more stable target. This prevents the network
from overfitting on immediately information, training on a diverse array of past
experiences. In implementation, the experience replay buffer is a fixed-size
FIFO queue.


### Seperate Target networks

DQNs also use a separate network to train target Q-value which will be used to
compute the loss for every action during training. While this seems like an
extra layer of indirection, deep neural networks do a lot of parameter tuning
while training which may cause a single network to fall into feedback loops
while it tries to adjust for both target Q-values as well as estimated Q-values.
To prevent these oscillations, the target q-network is split out and updated
less frequently than our main value network.


### Double DQN

Double Q-learning is an algorithm from B&S which redirects the environment's
state to two value learners instead of one. Double-learning algorithms
are excellent at training faster, more reliably, and - as an added benefit -
they can combat the overfitting common in deep learning networks.

Note, however that overfitting seems to be acceptable in deep learning problems,
so long as it is uniformly overfitted. I'll have to learn more about what
constitutes the 'acceptable parameters' for this some other time.


    Q-Target = r + γQ(s’,argmax(Q(s’,a,ϴ),ϴ’)


### Dueling Architecture

In a dueling architecture, we take our Q-function and decompose it into two
seperate components each with its own network representation: value and
advantage estimation (`Q(s,a) = V(s) + A(a)`). A value function tells us how
much long-term (ideally) value we get by staying in the current state, while
the advantage function tells us how much value we will get by making an action.

In doing this, we decouple the concerns of state and movement -- allowing our
agent to accurately predict state without concern for movement, and then we
combine two in a final step to predict our Q-value. I haven't written out the
inverse situation (allowing us to predict movement without concern for state)
because I'm uncertain at the time of writing if that would be an advisable
thing to do: ideally you would still maintain a dependence on state when
predicting movement.

"""
