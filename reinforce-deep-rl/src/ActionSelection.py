from Zoo.Prelude import *

def greedy_choice(q_function, state):
    qs = q_function(state)
    return int(np.argmax(qs))

def random_choice(action_size):
    return np.random.randint(0, action_size)

def epsilon_greedy_choice(epsilon, action_size, q_function, state):
    if np.random.rand(1) < epsilon:
        return random_choice(action_size)
    else:
        return greedy_choice(q_function, state)

def softmax_choice(q_function, state, temp=1):
    """
    IE: the boltzmann distribution

    Keep in mind that this distribution is, semantically, the probability an
    action has at being an optimal action. It is not how certain the agent's
    belief is that the action is optimal.
    """
    qs    = q_function(state)
    expQs = np.exp(qs / temp)
    probs = expQs / np.sum(expQs)

    action_value = np.random.choice(qs, probs)
    return np.argmax(qs == action_value)

#def tf_bayesian_dropout(q_function, inputs, hidden, state, action_selector, keep=0.5):
#    """
#     What if an agent could exploit its own uncertainty about its actions? This
#     is exactly the ability that a class of neural network models referred to as
#     Bayesian Neural Networks (BNNs) provide. Unlike traditional neural network
#     which act deterministically, BNNs act probabilistically. This means that
#     instead of having a single set of fixed weights, a BNN maintains a
#     probability distribution over possible weights. In a reinforcement learning
#     setting, the distribution over weight values allows us to obtain
#     distributions over actions as well. The variance of this distribution
#     provides us an estimate of the agent’s uncertainty about each action.
#
#     In practice however it is impractical to maintain a distribution over all
#     weights. Instead we can utilize dropout to simulate a probabilistic
#     network. Dropout is a technique where network activations are randomly
#     set to zero during the training process in order to act as a regularizer.
#     By repeatedly sampling from a network with dropout, we are able to obtain
#     a measure of uncertainty for each action.
#
#     When taking a single sample from a network with Dropout, we are doing
#     something that approximates sampling from a BNN. For more on the
#     implications of using Dropout for BNNs, I highly recommend Yarin Gal’s
#     Phd thesis on the topic[1].
#
#     [1]: http://mlg.eng.cam.ac.uk/yarin/blog_2248.html
#
#     Shortcomings: In order to get true uncertainty estimates, multiple
#     samples are required, thus increasing computational complexity. In my own
#     experiments however I have found it sufficient to sample only once, and
#     use the noisy estimates provided by the network. In order to reduce the
#     noise in the estimate, the dropout keep probability is simply annealed
#     over time from 0.1 to 1.0.
#     """
#     qOut     = q_function
#     keep_per = tf.placeholder(shape=None, dtype=tf.float32)
#     hidden   = slim.dropout(hidden, keep_per)
#
#
#     qs = sess.run(qOut, feed_dict={inputs:[state], keep_per:keep})
#     action = action_selector(q_function, state, *args)

"""
All of the methods discussed above deal with the selection of actions. There
is another approach to exploration that deals with the nature of the reward
signal itself. These approaches fall under the umbrella of intrinsic motivatio
, and there has been a lot of great work in this area. In a future post I will
be exploring these approaches in more depth, but for those interested, here is
a small selection of notable recent papers on the topic:

+ Variational Information Maximizing Exploration
  => https://arxiv.org/abs/1605.09674
+ Incentivizing Exploration In Reinforcement Learning With Deep Predictive Models
  => https://arxiv.org/abs/1507.00814
+ Unifying Count-Based Exploration and Intrinsic Motivation
  => https://arxiv.org/abs/1606.01868
+ Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction and Intrinsic Motivation
  => https://arxiv.org/abs/1604.06057

See http://arxiv.org/pdf/1507.00814 for a more in-depth comparison that was done using different techniques.
"""
