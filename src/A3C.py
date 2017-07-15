from Zoo.Prelude import *

import threading
import multiprocessing

from random import choice
from time import sleep, time


def update_target_graph(from_scope, to_scope):
    """
    Copies one set of variables to another.
    Used to set worker network parameters to those of global network.
    """
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars   = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))

    return op_holder

def process_frame(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()


def discount(r, gamma):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 0:
        running_add = 0
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

class ActorCriticNetwork():
    class ImageInputNetwork:
        def __init__(self, state_size, xlen, ylen, chans):
            input_flat    = tf.placeholder(shape=[None, state_size], dtype=tf.float32)
            input_resized = tf.reshape(input_flat, shape=[-1, xlen, ylen, chans])

            conv1 = slim.conv2d(
                activation_fn=tf.nn.elu,
                inputs=input_resized,
                num_outputs=16,
                kernel_size=[8,8],
                stride=[4,4],
                padding='VALID')

            conv2 = slim.conv2d(
                activation_fn=tf.nn.elu,
                inputs=conv1,
                num_outputs=32,
                kernel_size=[4,4],
                stride=[2,2],
                padding='VALID')

            hidden = slim.fully_connected(slim.flatten(conv2), 256, activation_fn=tf.nn.elu)

    class RecurrentNetwork:
        def __init__(self, conv):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)

            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]

            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)

            rnn_in    = tf.expand_dims(hidden, [0])
            step_size = tf.shape(self.imageIn)[:1]
            state_in  = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)

            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell,
                rnn_in,
                initial_state=state_in,
                sequence_length=step_size,
                time_major=False)

            lstm_c, lstm_h = lstm_state

            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            self.output = tf.reshape(lstm_outputs, [-1, 256])

    def __init__(self, s_size, action_size, scope, trainer=None):
        with tf.variable_scope(scope):
            self.conv = self.ImageInputNetwork(state_size, xlen, ylen, chans)
            self.rnn  = self.RecurrentNetwork(self.conv)

            self.policy_net_output = slim.fully_connected(
                self.rnn.output,
                action_size,
                activation_fn=tf.nn.softmax,
                weights_initializer=normalized_columns_initializer(0.01),
                biases_initializer=None)

            self.value_net_output = slim.fully_connected(
                self.rnn.output,
                1,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None)

            # Only the worker network need ops for loss functions and gradient updating.
            if trainer is not None:
                self.compute_gradients(self.rnn, action_size, scope, trainer)

    def compute_gradients(self, rnn, action_size, scope, trainer):
        actions        = tf.placeholder(shape=[None], dtype=tf.int32)
        target_v       = tf.placeholder(shape=[None], dtype=tf.float32)
        advantages     = tf.placeholder(shape=[None], dtype=tf.float32)
        actions_onehot = tf.one_hot(actions, action_size, dtype=tf.float32)

        responsible_outputs = tf.reduce_sum(self.policy_net_output * actions_onehot, [1])

        value_loss  = 0.5 * tf.reduce_sum(tf.square(target_v - tf.reshape(self.value_net_output, [-1])))
        entropy     = -tf.reduce_sum(self.policy_net_output * tf.log(self.policy_net_output))
        policy_loss = -tf.reduce_sum(tf.log(responsible_outputs) * advantages)
        loss        = 0.5 * value_loss + policy_loss - entropy * 0.01

        # Get gradients from local network using local losses
        local_vars        = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        gradients         = tf.gradients(loss, local_vars)
        var_norms         = tf.global_norm(local_vars)
        grads, grad_norms = tf.clip_by_global_norm(gradients, 40.0)

        # send local gradients to global network
        global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
        apply_grads = trainer.apply_gradients(zip(grads, global_vars))


class WorkerAgent:
    def __init__(self, env, name, state_size, action_size, trainer, model_path, global_episodes):
        self.env                 = env
        self.name                = "worker_" + str(name)
        self.number              = name
        self.model_path          = model_path
        self.trainer             = trainer
        self.global_episodes     = global_episodes
        self.increment           = self.global_episodes.assign_add(1)
        self.episode_rewards     = []
        self.episode_lengths     = []
        self.episode_mean_values = []
        self.summary_writer      = tf.summary.FileWriter("train_"+str(self.number))

        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_network    = ActorCriticNetwork(state_size, action_size, name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)
        actions               = np.identity(action_size, dtype=bool).tolist()

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout           = np.array(rollout)
        observations      = rollout[:,0]
        actions           = rollout[:,1]
        rewards           = rollout[:,2]
        next_observations = rollout[:,3]
        values            = rollout[:,5]
        localnet          = self.local_network

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        rewards_plus       = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(rewards_plus, gamma)[:-1]
        value_plus         = np.asarray(values.tolist() + [bootstrap_value])
        advantages         = discount(rewards + gamma * value_plus[1:] - value_plus[:-1], gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        rnn_state = localnet.rnn.state_init

        v_l, p_l, e_l, g_n, v_n, _ = sess.run(
            [
                localnet.value_loss,
                localnet.policy_loss,
                localnet.entropy,
                localnet.grad_norms,
                localnet.var_norms,
                localnet.apply_grads
            ],
            feed_dict = {
                localnet.target_v: discounted_rewards,
                localnet.inputs: np.vstack(observations),
                localnet.actions: actions,
                localnet.advantages: advantages,
                localnet.state_in[0]: rnn_state[0],
                localnet.state_in[1]: rnn_state[1]
            })
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def run_learner(self, max_episode_length, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        localnet      = self.local_network
        total_steps   = 0

        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_reward = 0
                episode_step_count = 0
                done = False

                s         = process_frame(self.env.reset())
                rnn_state = localnet.state_init

                while not done:
                    #Take an action using probabilities from policy network output.
                    a_dist, v, rnn_state = sess.run(
                        [
                            localnet.policy,
                            localnet.value,
                            localnet.rnn.state_out
                        ], feed_dict={
                            localnet.inputs:[s],
                            localnet.rnn.state_in[0]:rnn_state[0],
                            localnet.rnn.state_in[1]:rnn_state[1]
                        })

                    a = np.random.choice(a_dist[0],p=a_dist[0])
                    a = np.argmax(a_dist == a)
                    if a == 0:
                        a = 0
                    elif a == 1:
                        a = 2
                    elif a == 2:
                        a = 3
                    else:
                        raise Exception('no action')

                    s1, r, done, _ = self.env.step(a)
                    s1 = process_frame(s1)


                    episode_buffer.append([s,a,r,s1,done,v[0,0]])
                    episode_values.append(v[0,0])

                    episode_reward     += r
                    s                   = s1
                    total_steps        += 1
                    episode_step_count += 1

                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if len(episode_buffer) == 30 and done != True and episode_step_count != max_episode_length - 1:
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        v1 = sess.run(
                            localnet.value,
                            feed_dict={
                                localnet.inputs:[s],
                                localnet.state_in[0]:rnn_state[0],
                                localnet.state_in[1]:rnn_state[1]
                            })[0,0]

                        v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # Update the network using the experience buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,0.0)

                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    if episode_count % 250 == 0 and self.name == 'worker_0':
                        saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
                        print ("Saved Model")

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value  = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)
                    self.summary_writer.flush()

                if self.name == 'worker_0':
                    sess.run(self.increment)

                episode_count += 1

if __name__ == '__main__':
    max_episode_length = 300
    gamma = .99 # discount rate for advantage estimation and reward discounting
    s_size = 80 * 80 * 1 # Observations are greyscale frames of 80 * 80 * 1
    action_size = 3 # Agent can move Left, Right, or Stay
    load_model = False
    model_path = './model'

    tf.reset_default_graph()

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    with tf.device("/cpu:0"):
        global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        trainer         = tf.train.AdamOptimizer(learning_rate=1e-4)
        primary_network = AC_Network(s_size,action_size,'global',None) # Generate global network
        num_workers     = multiprocessing.cpu_count() # Set workers ot number of available CPU threads
        new_worker      = lambda wid: \
            Worker(
                gym.make('Pong-v0'),
                wid,
                state_size,
                action_size,
                trainer,
                model_path,
                global_episodes)

        workers = [ new_worker(i) for i in range(num_workers) ]
        saver   = tf.train.Saver(max_to_keep=5)

    with tf.Session(config=sess_config) as sess:
        coord = tf.train.Coordinator()
        if load_model == True:
            print ('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess,ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        # This is where the asynchronous magic happens.
        # Start the "work" process for each worker in a separate threat.
        def runner(worker):
            t = threading.Thread(target=(lambda: worker.work(max_episode_length, gamma, sess, coord, saver)))
            t.start()
            sleep(0.5)
            return t

        worker_threads = [runner(worker) for worker in workers]

        coord.join(worker_threads)

