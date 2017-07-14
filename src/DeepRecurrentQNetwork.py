from Zoo.Prelude       import *
from Zoo.Gridworld     import Gridworld
from Zoo.BasicConvoNet import ConvolutionNetwork
from Zoo.ReplayBuffer  import ReplayBuffer

class DuelingRecurrentNetwork():
    class RecurrentLayer:
        def __init__(self, output, batch_size, train_length, hidden_size, rnn_cell, scope):
            """
            The input must be reshaped into [batch x trace x units] for rnn processing,
            and then returned to [batch x units] when sent through the upper levels.
            """
            conv_flat    = tf.reshape(slim.flatten(output), [batch_size, train_length, hidden_size])

            state_input = rnn_cell.zero_state(batch_size, tf.float32)

            rnn_out, state = tf.nn.dynamic_rnn(
                    inputs=conv_flat,
                    cell=rnn_cell,
                    dtype=tf.float32,
                    initial_state=state_input,
                    scope=scope+'_rnn'
                    )

            self.hidden_size  = hidden_size
            self.state_input  = state_input
            self.state        = state
            self.output       = tf.reshape(rnn_out, shape=[-1, hidden_size])
            self.train_length = train_length
            self.batch_size   = batch_size

        def reset_state(self, start):
            """ TODO """
            new_state = np.zeros([start, self.hidden_size])
            return (new_state, new_state)

        def run(self, sess, rnn_state, s):
            return sess.run(
                feed_dict={
                    self.scalar_input: [s/255.0],
                    self.train_length: 1,
                    self.state_input: rnn_state,
                    self.batch_size: 1
                })

    def __init__(self, xlen, ylen, chans, hidden_size, action_size, rnn_cell, scope):
        cnn          = ConvolutionNetwork(84, 84, 3, hidden_size)
        batch_size   = tf.placeholder(dtype=tf.int32)
        train_length = tf.placeholder(dtype=tf.int32)
        rnn          = self.RecurrentLayer(cnn.output, batch_size, train_length, hidden_size, rnn_cell, scope)

        # Take the output from the convolutional layer and send it to a recurrent layer.

        # The output from the recurrent player is then split into separate Value and Advantage streams
        advantageStream, valueStream = tf.split(rnn.output, num_or_size_splits=2, axis=1)

        xavier_init = tf.contrib.layers.xavier_initializer()
        advantageW  = tf.Variable(xavier_init([int(hidden_size / 2), action_size]))
        valueW      = tf.Variable(xavier_init([int(hidden_size / 2), 1]))

        value      = tf.matmul(valueStream    , valueW    )
        # ==============================================================================
        advantage0 = tf.matmul(advantageStream, advantageW)
        salience   = tf.gradients(advantage0, cnn.imageIn)   # what the hell is this???
        # ==============================================================================
        advantage  = tf.subtract(advantage0, tf.reduce_mean(advantage0, axis=1, keep_dims=True))
        qOut       = value + advantage
        predict    = tf.argmax(qOut, 1)

        # generate our q values and error
        targetQ  = tf.placeholder(shape=[None], dtype=tf.float32)
        actions  = tf.placeholder(shape=[None], dtype=tf.int32)
        qValues  = tf.reduce_sum(tf.multiply(qOut, tf.one_hot(actions, action_size, dtype=tf.float32)), axis=1)
        td_error = tf.square(targetQ - qValues)

        # To only propogate half of the error through the gradients, we will mask
        # the first half of our training data (see: Lample & Chatlot 2016)
        shape_half  = [batch_size, train_length//2]
        mask        = tf.reshape(tf.concat([tf.zeros(shape_half), tf.ones(shape_half)], 1), [-1])
        loss        = tf.reduce_mean(td_error * mask)
        updateModel = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

        """ things we need """
        self.scalar_input = cnn.scalar_input
        self.batch_size   = batch_size
        self.rnn          = rnn
        self.train_length = train_length
        self.predict      = predict
        self.qOut         = qOut
        self._targetQ     = targetQ
        self._actions     = actions
        self._updateModel = updateModel

    def update_network(self, sess, train_batch, targetQ, trace_length, state_train, batch_size):
        sess.run(
            self._updateModel,
            feed_dict={
                self.scalar_input: np.vstack(train_batch[:,0]/255.0),
                self._targetQ: targetQ,
                self._actions: train_batch[:,1],
                self.rnn.train_length: trace_length,
                self.rnn.state_input: state_train,
                self.rnn.batch_size: batch_size
            })


class Agent:
    def __init__(
            self,
            env,
            batch_size       = 4,         # How many experience traces to use for each training step.
            trace_length     = 8,         # How long each experience trace will be when training
            update_freq      = 5,         # How often to perform a training step.
            gamma            = 0.99,      # Discount factor on the target Q-values
            anneling_epsilon = (1.0, 0.1),# Start and end of epsilon for choosing random action
            anneling_steps   = 10000,     # How many steps of training to reduce startE to endE.
            max_episodes     = 10000,     # How many episodes of game environment to train network with.
            pre_train_steps  = 10000,     # How many steps of random actions before training begins.
            load_model       = False,     # Whether to load a saved model.
            path             = "./drqn",  # The path to save our model to.
            hidden_size      = 512,       # size of final convo-layer before splitting to advantage and value streams.
            max_steps        = 50,        # The max allowed length of our episode.
            time_per_step    = 1,         # Length of each step used in gif creation
            summary_length   = 100,       # Number of epidoes to periodically save for analysis
            tau              = 0.001      # Rate to update target network toward self.primary_network network
        ):
        self.learning_rate    = self.tau = tau
        self.env              = env
        self.batch_size       = batch_size
        self.update_freq      = update_freq
        self.gamma            = gamma
        self.trace_length     = trace_length
        self.max_episodes     = max_episodes
        self.pre_train_steps  = pre_train_steps
        self.max_steps        = max_steps
        self.load_model       = load_model
        self.path             = path
        self.summary_length   = summary_length
        self.eps              = \
                AnnealingEpsilon(
                    anneling_steps,
                    start_step=pre_train_steps,
                    eps_range=anneling_epsilon)

        self.xlen        = xlen        = 84
        self.ylen        = ylen        = 84
        self.chans       = chans       = 3
        self.action_size = action_size = 4

        self.experience  = ReplayBuffer(buffer_size=1000)

        tf.reset_default_graph()

        new_lstm   = lambda: tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)

        with tf.name_scope('primary_network'):
            self.primary_network = DuelingRecurrentNetwork(84, 84, 3, hidden_size, 4, new_lstm(), 'main')
        with tf.name_scope('target_network'):
            self.target_network  = DuelingRecurrentNetwork(84, 84, 3, hidden_size, 4, new_lstm(), 'target')

        # Make a path for our model to be saved in.
        if not os.path.exists(path):
            os.makedirs(path)

    def load(self, sess, saver):
        """ load the model to a session from a saver """
        if self.load_model == True:
            print('Loading Model..')
            ckpt = tf.train.get_checkpoint_state(self.path)
            saver.restore(sess, ckpt.model_checkpoint_path)

    def save(self, sess, i, saver):
        saver.save(sess, self.path+'/model-'+str(i)+'.cptk')
        print("Saved Model")

    def choose_action(self, eps, step_num, sess, network, state):
        """ choose an action epsilon-greedily """
        action_chooser = lambda: sess.run(network.predict, feed_dict={network.input: [state]})[0]
        return epsilon_greedy(eps, self.action_size, action_chooser, partial(self.in_pretraining, step_num))

    def in_pretraining(self, step):
        return step < self.pre_train_steps

    def run_learner(self):
        jList = []
        rList = []
        total_steps = 0

        rw         = ReportWriter()
        init       = tf.global_variables_initializer()
        trainables = tf.trainable_variables()
        targetOps  = update_target_graph(trainables, self.tau)
        primary    = self.primary_network
        target     = self.target_network
        saver      = tf.train.Saver(max_to_keep=5)

        with tf.Session() as sess:
            self.load(sess, saver)
            sess.run(init)
            update_target_net_to_primary_net(targetOps, sess)

            for i in range(self.max_episodes):
                ew = EpisodeWriter()
                episodeBuffer = []
                s = process_shape(self.env.reset())
                done = False
                rAll = 0
                j = 0

                #Reset the recurrent layer's hidden state
                rnn_state = primary.rnn.reset_state(1)

                #The Q-Network
                while j < self.max_steps and not done:
                    if done:
                        raise Exception()
                    j+=1
                    total_steps += 1

                    e = self.eps(total_steps)
                    if np.random.rand(1) < e or total_steps < self.pre_train_steps:
                        state1 = sess.run(primary.rnn.state,
                                feed_dict={
                                    primary.scalar_input: [s/255.0],
                                    primary.train_length: 1,
                                    primary.rnn.state_input: rnn_state,
                                    primary.batch_size: 1
                                })
                        a = np.random.randint(0,4)
                    else:
                        a, state1 = sess.run([
                                primary.predict,
                                primary.rnn.state
                                ],
                                feed_dict={
                                    primary.scalar_input: [s/255.0],
                                    primary.train_length: 1,
                                    primary.rnn.state_input: rnn_state,
                                    primary.batch_size: 1
                                })
                        a = a[0]

                    s1P,r,done = self.env.step(a)
                    s1 = process_shape(s1P)
                    episodeBuffer.append(np.reshape(np.array([s,a,r,s1,done]),[1,5]))
                    ew.tell(s,a,r,done)
                    # tellAll( advantages, state_value)

                    if total_steps > self.pre_train_steps:
                        if total_steps % self.update_freq == 0:
                            update_target_net_to_primary_net(targetOps, sess)
                            #Reset the recurrent layer's hidden state
                            state_train = primary.rnn.reset_state(self.batch_size)

                            #Get a random batch of experiences.
                            train_batch = self.experience.sample_sequence(self.batch_size, self.trace_length)

                            #Below we perform the Double-DQN update to the self.target_network Q-values
                            Q1 = sess.run(
                                    primary.predict,
                                    feed_dict={
                                        primary.scalar_input:np.vstack(train_batch[:,3]/255.0),
                                        primary.rnn.train_length:self.trace_length,
                                        primary.rnn.state_input:state_train,
                                        primary.rnn.batch_size:self.batch_size
                                    })
                            Q2 = sess.run(
                                    target.qOut,
                                    feed_dict={
                                        target.scalar_input: np.vstack(train_batch[:,3]/255.0),
                                        target.rnn.train_length:self.trace_length,
                                        target.rnn.state_input:state_train,
                                        target.rnn.batch_size:self.batch_size
                                    })

                            end_multiplier = -(train_batch[:,4] - 1)
                            doubleQ = Q2[range(self.batch_size*self.trace_length),Q1]
                            targetQ = train_batch[:,2] + (self.gamma*doubleQ * end_multiplier)

                            #Update the network with our target values.
                            primary.update_network(
                                    sess,
                                    train_batch,
                                    targetQ,
                                    self.trace_length,
                                    state_train,
                                    self.batch_size)
                    rAll += r
                    s = s1
                    rnn_state = state1


                #Add the episode to the experience buffer
                self.experience.add_episode(lzip(np.array(episodeBuffer))))
                jList.append(j)
                rList.append(rAll)

                #Periodically save the model.
                if i % 1000 == 0 and i != 0:
                    self.save(sess, i, saver)
                    print ("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")
                if len(rList) % self.summary_length == 0 and len(rList) != 0:
                    print(total_steps, np.mean(rList[-self.summary_length:]), e)
                    # saveToCenter(i,rList,jList,np.reshape(np.array(episodeBuffer),[len(episodeBuffer),5]),\
                    #    self.summary_length,hidden_size,sess,self.primary_network,time_per_step)

            self.save(sess, i, saver)
        print ("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")



def process_shape(s):
    """ hard-coded reshape to flatten a games' frames. """
    return np.reshape(s, [84*84*3])

# Set the target network to be equal to the self.primary_network network.
def update_target_graph(tf_vars, tau):
    """
    These functions allows us to update the parameters of our target network
    with those of the self.primary_network network.
    """
    total_vars      = len(tf_vars)
    half_total_vars = int(total_vars / 2)
    op_holder       = []

    for idx,var in enumerate(tf_vars[0:half_total_vars]):
        i = idx + half_total_vars
        q = (tau * var.value()) + ((1 - tau) * tf_vars[i].value())
        h = tf_vars[i].assign(q)
        op_holder.append(h)

    return op_holder


def update_target_net_to_primary_net(op_holder, sess):
    for op in op_holder:
        sess.run(op)

    half_size = int(len(tf.trainable_variables()) / 2)
    eval_at = lambda i: tf.trainable_variables()[i].eval(session=sess)

    if not eval_at(0).all() == eval_at(half_size).all():
        raise Exception("Target Set Failed")


if __name__ == '__main__':
    agent = Agent(env=Gridworld(partial=False,sizeX=9))
    agent.run_learner()

"""
According to the author, RNNs are particularly suited to POMDPs. This is because
in real-world scenarios what we need is to give an agent the "capacity for
temporal integration of observations." - a very cool-sounding notion. I wonder
where that comes from. What this means is that a single observation isn't always
good enough to make a decision. Ways we know of doing this that we know of at the moment include
eligibility traces (but maybe not because they don't integrate over observation,
just reward) and experience replay (how are they connected to eligibility traces?).
POMDPs, especially, require this kind of property in an agent, because the more
we can observe in our environment, the better we are able to understand certain
phenomina. I like the example of field of vision and how you may not be able to
understand what location you are in when you get a snap shot, but by changing
the observation to a range of motion where you are able to turn around and see
the entire room, you get a more accurate understanding.
"""
"""
> Within the context of Reinforcement Learning, there are a number of possible
> ways to accomplish this temporal integration.

DeepMind originally does this by storing up a buffer of observations (four at a
time) which holds this information. You can image this as "storing a buffer of
observations" while RNNs are a way of "maintaining the experience of processed
observation"
"""
"""
> By utilizing a recurrent block in our network, we can pass the agent single
> frames of the environment, and the network will be able to change its output
> depending on the temporal pattern of observations it receives.

An LSTM block can choose to feed the hidden state back into itself depending on
its activation layer, thus acting as an augmentation which tells the network
what has come before.
"""

"""
=====================================
Alterations:

- All this takes is one change to our network - an LSTM cell between the
  output of our convo-net and the input into the value-advantage splits.

  -> We do this with tf.nn.dynamic_rnn run on tf.nn.rnn_cell.LSTMCell which is
  -> feed to the rnn node. We also need to slightly alter the training process
  -> in order to send an empty hidden state to our recurrent cell at the
     beginning of each sequence.

- because we want to replay experience with a temporal element, we can't
  just draw randomly from our buffer. Instead we need sample traces of a
  given length. By doing this we both retain our random sampling as well
  as ensure each trace of experiences actually follows from one another.
  -> I think _this_ is an eligibility trace?

- We will be using a technique developed by a group at CMU who recently
used DRQN (paper?) to train a neural network to play the first person
shooter game Doom. Instead of sending all the gradients backwards when
training their agent, they sent only the last half of the gradients for
a given trace.
  -> We can do this by masking the loss for the first half of each trace
     in a batch. They found it improved performance by only sending more
     meaningful information through the network.
"""

