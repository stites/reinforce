from Zoo.Prelude import *


class Agent:

    def __init__(
            self,
            env,
            max_episodes=2000,
            max_steps=99,
            batch_size=5,
            batch_size_model=5,
            hidden_size=8,
            lr=0.1,
            gamma=0.99,
            eps=0.99      # decay factor for RMSProp leaky sum of grad^2
            ):
        self.action_size, self.state_size = space_sizes(env)
        self.hidden_size = hidden_size
        self.lr = lr

        # """ Build the tensorflow graph """
        tf.reset_default_graph()

        self.policy_network()
        self.model_network()


    def policy_network(self):
        (ssize, asize, hsize) = self.state_size, self.action_size, self.hidden_size

        inputs = tf.placeholder(shape=[None, ssize], dtype=tf.float32, name="input_x")

        w1     = tf.get_variable("w1", shape=[ssize, hsize], initializer=tf.contrib.layers.xavier_initializer())
        layer1 = tf.nn.relu(tf.matmul(inputs, w1))
        w2     = tf.get_variable("w2", shape=[hsize, asize], initializer=tf.contrib.layers.xavier_initializer())
        output = tf.matmul(layer1, w2)

        probability = tf.nn.sigmoid(output)

        tvars = tf.trainable_variables()
        input_y = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="input_y")
        advantages = tf.placeholder(dtype=tf.float32, name="reward_signal")
        adam = tf.train.AdamOptimizer(learning_rate = self.lr)

        w1Grad = tf.placeholder(dtype=tf.float32, name="batch_grad1")
        w2Grad = tf.placeholder(dtype=tf.float32, name="batch_grad2")
        batchGrad = [w1Grad, w2Grad]

        loglikelihood = tf.log(input_y * (input_y - probability) + (1 - input_y) * (input_y + probability))
        loss          = -tf.reduce_mean(loglikelihood * advantages)
        newGrads      = tf.gradients(loss, tvars)
        updateGrads   = adam.apply_gradients(zip(batchGrad, tvars))

    def model_network(self):
        """TODO"""
        pass

    def run_learner(self):
        """TODO"""
        pass

    def _rollout_ep(self, s, eps, sess):
        """TODO"""
        pass

def main():
    # agent = Agent(gym.make('CartPole-v0'), max_episodes=2000, max_steps=199)
    # agent.run_learner()
    print(pd.DataFrame([0,1,2,3]).sum())

if __name__ == "__main__":
    main()




