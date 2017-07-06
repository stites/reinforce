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
            hidden_size_model=256,
            lr=0.1,
            gamma=0.99,
            eps=0.99      # decay factor for RMSProp leaky sum of grad^2
            ):
        self.action_size, self.state_size = space_sizes(env)

        self.env               = env
        self.max_episodes      = max_episodes
        self.max_steps         = max_steps
        self.batch_size        = batch_size
        self.batch_size_model  = batch_size_model
        self.hidden_size       = hidden_size
        self.hidden_size_model = hidden_size_model
        self.lr                = lr
        self.gamma             = gamma
        self.eps               = eps

        """ Build the tensorflow graph """
        tf.reset_default_graph()

        self.policy = self.policy_network()
        self.model  = self.model_network()


    def policy_network(self):
        (ssize, asize, hsize) = self.state_size, self.action_size, self.hidden_size

        inputs = tf.placeholder(shape=[None, ssize], dtype=tf.float32, name="input_x")

        w1     = tf.get_variable("w1", shape=[ssize, hsize], initializer=tf.contrib.layers.xavier_initializer())
        layer1 = tf.nn.relu(tf.matmul(inputs, w1))
        w2     = tf.get_variable("w2", shape=[hsize, asize], initializer=tf.contrib.layers.xavier_initializer())
        output = tf.matmul(layer1, w2)

        probability = tf.nn.sigmoid(output)

        tvars      = tf.trainable_variables()
        input_y    = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="input_y")
        advantages = tf.placeholder(dtype=tf.float32, name="reward_signal")
        adam       = tf.train.AdamOptimizer(learning_rate=self.lr)

        w1Grad = tf.placeholder(dtype=tf.float32, name="batch_grad1")
        w2Grad = tf.placeholder(dtype=tf.float32, name="batch_grad2")
        batchGrad = [w1Grad, w2Grad]

        loglikelihood = tf.log(input_y * (input_y - probability) + (1 - input_y) * (input_y + probability))
        loss          = -tf.reduce_mean(loglikelihood * advantages)
        newGrads      = tf.gradients(loss, tvars)
        updateGrads   = adam.apply_gradients(zip(batchGrad, tvars))

        #==================================#
        # Things we want to refer to later #
        #==================================#
        return {
            "tvars": tvars
        }


    def model_network(self):
        (ssize, asize, hsize) = self.state_size, self.action_size, self.hidden_size_model

        # inputs are the current state and action
        ##inputs = tf.placeholder(dtype=tf.float32, shape=[None,asize+ssize])

        ## No idea what this is supposed to be
        ##with tf.variable_scope('rnnlm'):
        ##    softmax_w = tf.get_variable(name="softmax_weights", shape=[hsize, 50])
        ##    softmax_b = tf.get_variable(name="softmax_bias",    shape=[50])

        # we also track the previous input here
        previous_state = tf.placeholder(name="previous_state",  shape=[ None, asize+ssize], dtype=tf.float32)

        # start our first hidden layer of our model's neural network
        w1m     = tf.get_variable(name="w1m", shape=[asize+ssize,hsize], initializer=tf.contrib.layers.xavier_initializer())
        b1m     = tf.Variable(tf.zeros([hsize]), name="b1m")
        layer1m = tf.nn.relu(tf.matmul(previous_state, w1m) + b1m)

        # and again with our second hidden layer, which is locked in at the same size as our first layer.
        w2m     = tf.get_variable(name="w2m", shape=[hsize, hsize], initializer=tf.contrib.layers.xavier_initializer())
        b2m     = tf.Variable(tf.zeros([hsize]), name="b2m")
        layer2m = tf.nn.relu(tf.matmul(layer1m, w2m) + b2m)

        # Next we map different outputs depending on what we want out of our model. Each output layer has a
        # tensor of weights and biases, and we must also store a tensor of true values to approximate. The
        # output layers includes a layer fo state, reward, and completion.

        # An output layer for the observed state
        wObservation          = tf.get_variable(shape=[hsize, 4], name="wO", initializer=tf.contrib.layers.xavier_initializer())
        bObservation          = tf.Variable(tf.zeros([4]), name="bO")
        predicted_observation = tf.matmul(layer2m, wObservation, name="predicted_observation") + bObservation
        true_observation      = tf.placeholder(dtype=tf.float32, shape=[None,4], name="true_observation")
        # talk a bit more about this loss function
        observation_loss      = tf.square(true_observation - predicted_observation)

        # An estimated reward
        wReward          = tf.get_variable(shape=[hsize, 1], name="wR", initializer=tf.contrib.layers.xavier_initializer())
        bReward          = tf.Variable(tf.zeros([1]), name="bR")
        predicted_reward = tf.matmul(layer2m, wReward, name="predicted_reward") + bReward
        true_reward      = tf.placeholder(dtype=tf.float32, shape=[None,1], name="true_reward")
        # same loss as with observation
        reward_loss      = tf.square(true_reward - predicted_reward)

        # and a boolean of whether or not the episode has completed.
        wDone          = tf.get_variable(shape=[hsize, 1], name="wD", initializer=tf.contrib.layers.xavier_initializer())
        bDone          = tf.Variable(tf.ones([1]),  name="bD")
        predicted_done = tf.matmul(layer2m, wDone, name="predicted_done") + bDone
        true_done      = tf.placeholder(dtype=tf.float32, shape=[None,1], name="true_done")
        # talk a bit more about this loss function
        done_loss      = -tf.log(tf.multiply(predicted_done, true_done) + tf.multiply(1 - predicted_done, 1 - true_done))

        # the final output is actually simple the concatenation of all predicted outcomes
        predicted_state = tf.concat([predicted_observation, predicted_reward, predicted_done], axis=1)

        # and we want to minimize the aggregate loss across all predicted outcomes
        model_loss      = tf.reduce_mean(observation_loss + done_loss + reward_loss)
        update_model    = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(model_loss)

        return {
            "previous_state" : previous_state,
            "predicted_state": predicted_state,
            "model_loss"     : model_loss,
            "update_model"   : update_model
        }

    def step_model(sess, xs, action):
        """this function uses our model to produce a new state when given a previous state and action"""
        feed             = np.reshape(np.hstack([xs[-1][0], np.array(action)]), [1,5])
        predicted_model  = sess.run([self.model['predicted_state']], feed_dict={self.model['previous_state']:feed})[0]
        reward           = npmap(_4, predicted_model)
        observation      = predicted_model[:,0:4]
        observation[:,0] = np.clip(observation[:,0], -2.4, 2.4)
        observation[:,2] = np.clip(observation[:,2], -0.4, 0.4)
        done             = np.clip(predicted_model[:,5], 0, 1) > 0.1 or len(xs) > 300
        return observation, reward, done


    def run_learner(self):
        episodes = Writer()
        policy = self.policy
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            gradBuffer = lmap(mul(0), sess.run(policy.tvars)) # self.tvars should really be all tvars under a certain scope
            self._rollout_ep(self.env.reset(), None, sess)
            for ep_num in range(self.max_episodes):
                episodes.tell(self._rollout_ep(self.env.reset(), self.eps, sess))
                model_prob = sess.run(probability)

    def _rollout_ep(self, s, eps, sess):
        history = EpisodeWriter()
        
        pass


def main():
    # agent = Agent(gym.make('CartPole-v0'), max_episodes=2000, max_steps=199)
    # agent.run_learner()
    pass

if __name__ == "__main__":
    main()




