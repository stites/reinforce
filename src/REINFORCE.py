import gym
import operator
import numpy       as np
import tensorflow  as tf
import tensorflow.contrib.slim as slim


class Agent:

    def __init__(self, env, max_episodes=2000, max_steps=99, batch_size=5, hidden_size=8, lr=0.1, gamma=0.99):
        action_size = env.action_space.n
        state_size = env.observation_space.shape[0]

        # """ Build the tensorflow graph """
        tf.reset_default_graph()

        inputs  = tf.placeholder(shape = [None, state_size], dtype = tf.float32)
        hidden  = slim.fully_connected(inputs, hidden_size, biases_initializer=None, activation_fn=tf.nn.relu)
        outputs = slim.fully_connected(hidden, action_size, biases_initializer=None, activation_fn=tf.nn.softmax)

        reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

        indexes             = tf.range(0, tf.shape(outputs)[0]) * tf.shape(outputs)[1] + action_holder
        responsible_outputs = tf.gather(tf.reshape(outputs, [-1]), indexes)
        loss                = -tf.reduce_mean(tf.log(responsible_outputs) * reward_holder)

        tvars = tf.trainable_variables()

        def new_grad(idx, grad):
            return tf.placeholder(tf.float32, name=str(idx)+"_holder", shape=grad.shape)

        gradient_holders = imap(new_grad, tvars)
        gradients        = tf.gradients(loss, tvars)
        update_batch     = tf.train.AdamOptimizer(learning_rate = lr).apply_gradients(zip(gradient_holders,tvars))


        self.reward_holder = reward_holder
        self.action_holder = action_holder
        self.inputs        = inputs
        self.outputs       = outputs
        self.indexes       = indexes
        self.loss          = loss
        self.responsible_outputs = responsible_outputs
        self.gradient_holders    = gradient_holders
        self.gradients           = gradients
        self.update_batch        = update_batch

        self.gamma, self.max_steps = gamma, max_steps
        self.env, self.max_episodes, self.batch_size = env, max_episodes, batch_size



    def run_learner(self):
        _rollout_ep = self._rollout_ep
        env, max_episodes, batch_size = self.env, self.max_episodes, self.batch_size

        init = tf.global_variables_initializer()

        # Launch the tensorflow graph
        with tf.Session() as sess:
            sess.run(init)
            i = 0

            gradBuffer = reset_grads(sess.run(tf.trainable_variables()))

            stList = []
            rwdList = []

            while i < max_episodes:
                running_reward, nsteps, grads = _rollout_ep(env.reset(), None, sess)
                rwdList.append(running_reward)
                stList.append(nsteps)
                gradBuffer = list(map(lambda gs: gs[0] + gs[1], zip(grads, gradBuffer)))

                if i % 100 == 0:
                    print(np.mean(rwdList[-100:]))

                if i % batch_size == 0 and i != 0:
                    feed_dict= dictionary = dict(zip(self.gradient_holders, gradBuffer))
                    _ = sess.run(self.update_batch, feed_dict=feed_dict)
                    gradBuffer = reset_grads(gradBuffer)

                i += 1

        return stList, rwdList


    def _rollout_ep(self, s, eps, sess):
        running_reward = 0
        step_num = 0
        done = False
        ep_history = []

        while step_num < self.max_steps and not done:
            action_dist        = sess.run(self.outputs, feed_dict={self.inputs:[s]})
            action             = choose_action(action_dist, action_dist[0])
            s_next, rwd, done, _ = self.env.step(action)
            ep_history.append([s, action, rwd])

            # Book-keeping
            step_num += 1
            running_reward += rwd
            s = s_next

        #Update the network.
        grads = sess.run(self.gradients, feed_dict={
                self.inputs: _c(np.vstack, np.array, _w1(lmap, _0))(ep_history),
                self.action_holder: lmap(_1, ep_history),
                self.reward_holder: eligibility_trace(self.gamma, lmap(_2, ep_history))
            })

        return running_reward, step_num, grads

def compose(*functions):
    """ compose """
    def inner(arg):
        for f in reversed(functions):
            arg = f(arg)
        return arg
    return inner

_c = compose

def curry1(fn, a):
    def inner(*args):
        return fn(a, *args)
    return inner

_w1 = curry1

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

def decay_epsilon(ep_number):
    """ Reduce chance of random action as we train the model. """
    return 1.0 / ((ep_number / 50) + 10)

def reset_grads(gradBuffer):
    return list(map(lambda grad: grad * 0, gradBuffer))

def one_hot_encode(i, total):
  return np.identity(total)[i:i+1]

def main():
    agent = Agent(gym.make('CartPole-v0'), max_episodes=2000, max_steps=199)
    agent.run_learner()


if __name__ == "__main__":
    main()


