from Zoo.Prelude   import *
from Zoo.Gridworld import Gridworld, gameEnv
import os
import random

class ReplayBuffer:
    def __init__(self, buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, s, a, r, s1, done):
        self.append(np.reshape(np.array([s,a,r,s1,done]),[1,5]))

    def append(self, experience):
        lqueue = len(self.buffer)
        lexp   = len(experience)
        if lqueue + lexp >= self.buffer_size:
            self.buffer[0:(lqueue + lexp - self.buffer_size)] = []
        self.buffer.extend(experience)


    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])


class DuelingNetwork:
    class ConvolutionNetwork:
        def __init__(self, xlen, ylen, chans, hidden_size):
            flatten_flag = -1

            scalar_input = tf.placeholder(shape=[None, xlen*ylen*chans], dtype=tf.float32)
            imageIn      = tf.reshape(scalar_input, shape=[flatten_flag, xlen, ylen, chans])

            conv1 = slim.conv2d(
                biases_initializer=None,
                inputs=imageIn,          # input layer
                num_outputs=32,          # # of filters to apply to the previous layer
                kernel_size=[8,8],       # window size to slide over the previous layer
                stride=[4,4],            # pixels to skip as we move the window across the layer
                padding='VALID')         # if we want the window to slide over only the bottom
                                         # layer ("VALID") or add padding around it ("SAME") to
                                         # ensure that the convolutional layer has the same
                                         # dimensions as the previous layer.
            conv2 = slim.conv2d(
                inputs=conv1,
                num_outputs=64,
                kernel_size=[4,4],
                stride=[2,2],
                padding='VALID',
                biases_initializer=None)

            conv3 = slim.conv2d(
                inputs=conv2,
                num_outputs=64,
                kernel_size=[3,3],
                stride=[1,1],
                padding='VALID',
                biases_initializer=None)

            conv4 = slim.conv2d(
                inputs=conv3,
                num_outputs=hidden_size,
                kernel_size=[7,7],
                stride=[1,1],
                padding='VALID',
                biases_initializer=None)

            self.scalar_input = scalar_input
            self.output       = conv4

    def __init__(self, xlen, ylen, chans, hidden_size, action_size):
        self.cnn = self.ConvolutionNetwork(xlen, ylen, chans, hidden_size)
        self.input = self.cnn.scalar_input

        # We take the output from the final convolutional layer and split
        # it into separate advantage and value streams.
        advantageStreamConv, valueStreamConv = tf.split(self.cnn.output, 2, 3)
        advantageStream = slim.flatten(advantageStreamConv)
        valueStream     = slim.flatten(valueStreamConv)

        xavier_init = tf.contrib.layers.xavier_initializer()
        advantageW  = tf.Variable(xavier_init([int(hidden_size / 2), action_size]))
        valueW      = tf.Variable(xavier_init([int(hidden_size / 2), 1]))

        value     = tf.matmul(valueStream    , valueW    )
        advantage = tf.matmul(advantageStream, advantageW)
        advantage = tf.subtract(advantage, tf.reduce_mean(advantage, axis=1, keep_dims=True))

        qOut    = value + advantage
        predict = tf.argmax(qOut, 1)

        # Below we obtain the loss by taking the sum of squares difference
        # between the target and prediction Q values.
        targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        actions = tf.placeholder(shape=[None], dtype=tf.int32)
        qValues = tf.reduce_sum(tf.multiply(qOut, tf_one_hot(actions, action_size)), axis=1)

        loss        = tf.reduce_mean(tf.square(targetQ - qValues))
        updateModel = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

        """ things we need """
        self.qOut        = qOut
        self.predict     = predict
        self.targetQ     = targetQ
        self.actions     = actions
        self.updateModel = updateModel


def tf_one_hot(ph, size, dtype=tf.float32):
    return tf.one_hot(ph, size, dtype=dtype)


class Agent:
    def __init__(
            self,
            env,
            batch_size       = 32,       # How many experiences to use for each training step.
            update_freq      = 4,        # How often to perform a training step.
            gamma            = 0.99,     # Discount factor on the target Q-values
            anneling_epsilon = (1.0, 0.1), # Start and end of epsilon for choosing random action
            anneling_steps   = 10000,    # How many steps of training to reduce startE to endE.
            max_episodes     = 10000,    # How many episodes of game environment to train network with.
            pre_train_steps  = 10000,    # How many steps of random actions before training begins.
            max_steps        = 50,       # The max allowed length of our episode.
            load_model       = False,    # Whether to load a saved model.
            path             = "./dqn",  # The path to save our model to.
            hidden_size      = 512,      # The size of the final convolutional layer before splitting
                                         # it into Advantage and Value streams.
            tau              = 0.001     # Rate to update target network toward primary network
        ):
        self.learning_rate    = self.tau = tau
        self.env              = env
        self.batch_size       = batch_size
        self.update_freq      = update_freq
        self.gamma            = gamma
        self.max_episodes     = max_episodes
        self.pre_train_steps  = pre_train_steps
        self.max_steps        = max_steps
        self.load_model       = load_model
        self.path             = path
        self.eps              = \
                AnnealingEpsilon(
                    anneling_steps,
                    start_step=pre_train_steps,
                    eps_range=anneling_epsilon)

        self.xlen        = xlen        = 84
        self.ylen        = ylen        = 84
        self.chans       = chans       = 3
        self.action_size = action_size = 4

        self.experience = ReplayBuffer()

        tf.reset_default_graph()

        with tf.name_scope('primary_network'):
          self.primary_network = DuelingNetwork(xlen, ylen, chans, hidden_size, action_size)

        with tf.name_scope('target_network'):
          self.target_network = DuelingNetwork(xlen, ylen, chans, hidden_size, action_size)

        if not os.path.exists(path):
            os.makedirs(path)

    def process_state(self, states):
        """ flatten a state from [xlen, ylen, chans] to [xlen * ylen * chans] """
        return np.reshape(states, [self.xlen * self.ylen * self.chans])

    def update_target_graph(self, tf_vars, tau):
        """ TODO """
        total_vars      = len(tf_vars)
        half_total_vars = int(total_vars / 2)
        op_holder       = []

        for idx,var in enumerate(tf_vars[0:half_total_vars]):
            i = idx + half_total_vars
            q = (self.tau * var.value()) + ((1 - self.tau) * tf_vars[i].value())
            h = tf_vars[i].assign(q)
            op_holder.append(h)

        return op_holder


    def update_target(self, op_holder, sess):
        """ TODO """
        for op in op_holder:
            sess.run(op)


    def load(self, sess, saver):
        """ load the model to a session from a saver """
        if self.load_model == True:
            print('Loading Model..')
            ckpt = tf.train.get_checkpoint_state(self.path)
            saver.restore(sess, ckpt.model_checkpoint_path)

    def save(self, sess, saver, i):
        saver.save(sess, self.path+'/model-'+str(i)+'.cptk')
        print("Saved Model")

    def choose_action(self, eps, step_num, sess, network, state):
        """ choose an action epsilon-greedily """
        action_chooser = lambda: sess.run(network.predict, feed_dict={network.input: [state]})[0]
        return epsilon_greedy(eps, self.action_size, action_chooser, partial(self.in_pretraining, step_num))

    def in_pretraining(self, step):
        return step < self.pre_train_steps

    def run_learner(self):
        primary_net = self.primary_network
        target_net  = self.target_network

        init         = tf.global_variables_initializer()
        saver        = tf.train.Saver()
        trainables   = tf.trainable_variables()
        targetOps    = self.update_target_graph(trainables, self.tau)

        # create lists to contain total rewards and steps per episode
        jList    = []
        rList    = []
        all_steps = 0
        epsilon = self.eps.start


        with tf.Session() as sess:
            sess.run(init)
            self.load(sess, saver)
            for ep_num in range(self.max_episodes):
                episode_experience = ReplayBuffer()
                # Reset environment and get first new observation
                s        = self.process_state(self.env.reset())
                done     = False
                rAll     = 0
                step_num = 0

                while step_num < self.max_steps and not done:
                    step_num += 1
                    all_steps += 1

                    epsilon         = self.eps(all_steps)
                    action          = self.choose_action(epsilon, step_num, sess, primary_net, s)
                    next_s, r, done = self.env.step(action)
                    next_s          = self.process_state(next_s)
                    #Save the experience to our episode buffer.
                    episode_experience.add(s,action,r,next_s,done)

                    if not self.in_pretraining(all_steps) and step_num % self.update_freq == 0:
                        training_batch = self.experience.sample(self.batch_size)
                        # Below we perform the Double-DQN update to the target Q-values
                        Q1 = sess.run(
                                primary_net.predict,
                                feed_dict={
                                    primary_net.input: np.vstack(training_batch[:,3])
                                })

                        Q2 = sess.run(
                                target_net.qOut,
                                feed_dict={
                                    target_net.input: np.vstack(training_batch[:,3])
                                })

                        end_multiplier = -(training_batch[:,4] - 1)
                        doubleQ = Q2[range(self.batch_size), Q1]
                        targetQ = training_batch[:,2] + (self.gamma * doubleQ * end_multiplier)

                        # Update the network with our target values.
                        _ = sess.run(
                                primary_net.updateModel,
                                feed_dict={
                                    primary_net.input  : np.vstack(training_batch[:,0]),
                                    primary_net.targetQ: targetQ,
                                    primary_net.actions: training_batch[:,1]
                                })

                        # Update the target network toward the primary network.
                        self.update_target(targetOps, sess)
                    rAll += r
                    s = next_s

                self.experience.append(episode_experience.buffer)
                jList.append(all_steps)
                rList.append(rAll)

                #Periodically save the model.
                if ep_num % 1000 == 0:
                    self.save(sess, saver, ep_num)
                    print("Percent of succesful episodes: " + str(sum(rList)/self.max_episodes) + "%")

                if len(rList) % 10 == 0:
                    print(ep_num, all_steps, np.mean(rList[-10:]), epsilon)

            self.save(sess, saver, ep_num)
            print("Percent of succesful episodes: " + str(sum(rList)/self.max_episodes) + "%")

if __name__ == "__main__":
    agent = Agent(Gridworld(partial=False, sizeX=5), load_model=True)
    agent.run_learner()

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


