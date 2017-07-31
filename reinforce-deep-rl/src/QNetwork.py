from Zoo.Prelude import *

class Agent:

    def __init__(self, env, max_episodes=2000, max_steps=99, lr=0.1, gamma=0.99):
        tf.reset_default_graph()

        """
        Qout[1,4] = inputs[1,16] * W[16,4]
        """
        inputs  = tf.placeholder(shape = [1, 16], dtype = tf.float32)
        W       = tf.Variable(tf.random_uniform([16, 4], 0, 0.01))
        Qout    = tf.matmul(inputs, W)

        predict = tf.argmax(Qout, 1) # shape is [1,1]


        """
        loss[1,4] = reduce_sum $ (nextQ[1,4] - Qout[1,4]) ^ 2
        """
        # loss is defined by the sum of squares difference between the target and prediction Q values.
        nextQ       = tf.placeholder(shape = [1,4], dtype = tf.float32)
        loss        = tf.reduce_sum(tf.square(nextQ - Qout))

        """
        trainer and model updater
        """
        trainer     = tf.train.GradientDescentOptimizer(learning_rate = lr)
        updateModel = trainer.minimize(loss)

        self.all = (env, inputs, W, Qout, predict, nextQ, loss, trainer, updateModel)
        self.env = env
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.lr = lr
        self.gamma = gamma


    def run_learner(self):
        env, max_episodes, _rollout_ep = self.env, self.max_episodes, self._rollout_ep


        stList = []
        rwdList = []

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            for i in range(max_episodes):
                step_total, reward_total = _rollout_ep(
                        s=env.reset(),
                        eps=decay_epsilon(i),
                        sess=sess)

                stList.append(step_total)
                rwdList.append(reward_total)

        print("Percent of succesful episodes: " + str(sum(rwdList)/max_episodes) + "%")
        print("Percent of succesful last 50: " + str(sum(rwdList[-50:])/50) + "%")

        return stList, rwdList


    def _rollout_ep(self, s, eps, sess):
        env, inputs, W, Qout, predict, nextQ, loss, trainer, updateModel = self.all
        gamma, max_steps = self.gamma, self.max_steps

        total_rewards = 0
        step_num = 0
        done = False
        one_hot = lambda s: one_hot_encode(s, total=16)

        while step_num < max_steps and not done:
            # Choose an action by greedily (with eps chance of random action) from the Q-network
            """ greedy_a = predict :: int = tf.argmax(Qout::16x4, 1::int)"""
            """ curQs    = Qout    :: 1x4 = ( inputs :: 1x16, PH ) * ( W :: 16x4, Var ) """
            greedy_a, curQs = sess.run([predict, Qout], feed_dict = { inputs:one_hot(s) })

            a = choose_action(eps, greedy_a, env.action_space)  #: int

            #Get new state and reward from environment
            s_next, r, done, _ = env.step(a)

            # Obtain the Q values by feeding the new state through our network
            Q_next = sess.run(Qout, feed_dict={ inputs:one_hot(s_next) })

            # Obtain maxQ' and set our target value for chosen action.
            maxQ_next = np.max(Q_next)
            targetQ = curQs
            targetQ[0, a] = r + gamma * maxQ_next

            # Train our network using target and predicted Q values. Notice that that the tensor is mutated
            # and requires no use of it's output
            _, _ = sess.run([updateModel, W],feed_dict={ inputs:one_hot(s), nextQ:targetQ })

            # Book-keeping
            step_num += 1
            total_rewards += r
            s = s_next

        return step_num, total_rewards


def decay_epsilon(ep_number):
    """ Reduce chance of random action as we train the model. """
    return 1.0 / ((ep_number / 50) + 10)


def choose_action(eps, greedy_a, action_space):
  if np.random.rand(1) < eps:
    return action_space.sample()
  else:
    return greedy_a[0]


def one_hot_encode(i, total):
  return np.identity(total)[i:i+1]


def main():
    agent = Agent(gym.make('FrozenLake-v0'), max_episodes=2000, max_steps=199)
    agent.run_learner()


if __name__ == "__main__":
    main()
