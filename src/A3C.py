from Zoo.Prelude import *

import threading
import multiprocessing

from random import choice
from time import sleep, time

GLOBAL_SCOPE = 'global'

# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.


def update_target_graph(from_scope, to_scope):
    from_vars  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars    = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    assign_var = lambda fromv, tov: tov.assign(fromv)

    return lmap(curry(assign_var), zip(from_vars, to_vars))


def process_pong(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


def discount(rs, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(rs)
    running_add = 0
    for t in reversed(range(0, rs.size)):
        if rs[t] != 0:
            running_add = 0
        running_add = running_add * gamma + rs[t]
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
    def __init__(self, state_size, action_size, scope, xlen, ylen, chans, trainer=None):
        with tf.variable_scope(scope.wid):
            self.conv = self.ImageInputNetwork(state_size, xlen, ylen, chans)
            self.rnn = self.RecurrentNetwork(self.conv)

            self.policy = slim.fully_connected(
                self.rnn.output,
                action_size,
                activation_fn=tf.nn.softmax,
                weights_initializer=normalized_columns_initializer(0.01),
                biases_initializer=None)

            self.value = slim.fully_connected(
                self.rnn.output,
                1,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None)

            # Only the worker network need ops for loss functions and gradient updating.
            if trainer is not None:
                self.compute_gradients(self.rnn, action_size, scope, trainer)

    def compute_gradients(self, rnn, action_size, scope, trainer):
        actions = tf.placeholder(shape=[None], dtype=tf.int32)
        target_v = tf.placeholder(shape=[None], dtype=tf.float32)
        advantages = tf.placeholder(shape=[None], dtype=tf.float32)
        actions_onehot = tf.one_hot(actions, action_size, dtype=tf.float32)

        responsible_outputs = tf.reduce_sum(self.policy * actions_onehot, [1])

        value_loss = self.value_loss = 0.5 * \
            tf.reduce_sum(tf.square(target_v - tf.reshape(self.value, [-1])))
        entropy = self.entropy = - \
            tf.reduce_sum(self.policy * tf.log(self.policy))
        policy_loss = self.policy_loss = - \
            tf.reduce_sum(tf.log(responsible_outputs) * advantages)
        loss = self.loss = 0.5 * value_loss + policy_loss - entropy * 0.01

        # Get gradients from local network using local losses
        local_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope.wid)
        gradients = tf.gradients(loss, local_vars)
        var_norms = tf.global_norm(local_vars)
        grads, grad_norms = tf.clip_by_global_norm(gradients, 40.0)

        # send local gradients to global network
        global_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
        apply_grads = trainer.apply_gradients(zip(grads, global_vars))

        self.advantages = advantages
        self.grad_norms = grad_norms
        self.var_norms = var_norms
        self.apply_grads = apply_grads
        self.target_v = target_v
        self.actions = actions


class ActorCriticNetwork():
    class ImageInputNetwork:
        def __init__(self, state_size, xlen, ylen, chans):
            self.input_raw     = tf.placeholder(shape=[None, state_size], dtype=tf.float32)
            self.input_resized = tf.reshape(self.input_raw, shape=[-1, xlen, ylen, chans])

            conv1 = slim.conv2d(activation_fn=tf.nn.elu, num_outputs=16, kernel_size=[8, 8], stride=[4, 4], padding='VALID', inputs=self.input_resized)
            conv2 = slim.conv2d(activation_fn=tf.nn.elu, num_outputs=32, kernel_size=[4, 4], stride=[2, 2], padding='VALID', inputs=conv1)

            self.hidden = slim.fully_connected(slim.flatten(conv2), 256, activation_fn=tf.nn.elu)

    class RecurrentNetwork:
        def __init__(self, conv):
            lstm_cell       = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)
            c_init          = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init          = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]

            c_in          = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in          = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)

            rnn_in    = tf.expand_dims(conv.hidden, [0])
            step_size = tf.shape(conv.input_resized)[:1]
            state_in  = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)

            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell,
                rnn_in,
                initial_state=state_in,
                sequence_length=step_size,
                time_major=False)

            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            self.rnn_out   = tf.reshape(lstm_outputs, [-1, 256])


    def __init__(self, state_size, action_size, scope, trainer, xlen, ylen, chans):
        with tf.variable_scope(scope):
            conv = self.conv = self.ImageInputNetwork(state_size, xlen, ylen, chans)
            lstm = self.lstm = self.RecurrentNetwork(self.conv)

            self.policy = \
                slim.fully_connected(
                    lstm.rnn_out,
                    action_size,
                    activation_fn=tf.nn.softmax,
                    weights_initializer=normalized_columns_initializer(0.01),
                    biases_initializer=None)

            self.value = \
               slim.fully_connected(
                    lstm.rnn_out,
                    1,
                    activation_fn=None,
                    weights_initializer=normalized_columns_initializer(1.0),
                    biases_initializer=None)

            # Only the worker network need ops for loss functions and gradient updating.
            if trainer is not None:
                self.compute_gradients(self.lstm, action_size, scope, trainer)

    def compute_gradients(self, rnn, action_size, scope_name, trainer):
        actions        = tf.placeholder(shape=[None], dtype=tf.int32)
        target_v       = tf.placeholder(shape=[None], dtype=tf.float32)
        advantages     = tf.placeholder(shape=[None], dtype=tf.float32)
        actions_onehot = tf.one_hot(actions, action_size, dtype=tf.float32)

        responsible_outputs = tf.reduce_sum(self.policy * actions_onehot, [1])

        value_loss  = self.value_loss  = 0.5 * tf.reduce_sum(tf.square(target_v - tf.reshape(self.value, [-1])))
        entropy     = self.entropy     = -tf.reduce_sum(self.policy * tf.log(self.policy))
        policy_loss = self.policy_loss = -tf.reduce_sum(tf.log(responsible_outputs) * advantages)
        loss        = self.loss        = 0.5 * value_loss + policy_loss - entropy * 0.01

        # Get gradients from local network using local losses
        local_vars        = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_name)
        gradients         = tf.gradients(loss, local_vars)
        var_norms         = tf.global_norm(local_vars)
        grads, grad_norms = tf.clip_by_global_norm(gradients, 40.0)

        # send local gradients to global network
        global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, GLOBAL_SCOPE)
        apply_grads = trainer.apply_gradients(zip(grads, global_vars))

        self.advantages = advantages
        self.grad_norms = grad_norms
        self.var_norms = var_norms
        self.apply_grads = apply_grads
        self.target_v = target_v
        self.actions = actions


class Worker():
    def __init__(self, env, wid, state_size, action_size, trainer, model_path, global_episodes):
        self.wid             = "worker_" + str(wid)
        self.number          = wid
        self.model_path      = model_path
        self.trainer         = trainer
        self.global_episodes = global_episodes
        self.increment       = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer      = tf.summary.FileWriter("train_" + str(wid))

        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = ActorCriticNetwork(state_size, action_size, self.wid, trainer, xlen, ylen, chans)
        self.update_local_ops = update_target_graph('global', self.wid)

        # The Below code is related to setting up the Doom environment
        self.actions = self.actions = np.identity(
            action_size, dtype=bool).tolist()
        # End Doom set-up
        self.env = env

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        next_observations = rollout[:, 3]
        values = rollout[:, 5]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * \
            self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        rnn_state = self.local_AC.lstm.state_init
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.conv.input_raw: np.vstack(observations),
                     self.local_AC.actions: actions,
                     self.local_AC.advantages: advantages,
                     self.local_AC.lstm.state_in[0]: rnn_state[0],
                     self.local_AC.lstm.state_in[1]: rnn_state[1]}
        v_l, p_l, e_l, g_n, v_n, _ = sess.run([self.local_AC.value_loss,
                                               self.local_AC.policy_loss,
                                               self.local_AC.entropy,
                                               self.local_AC.grad_norms,
                                               self.local_AC.var_norms,
                                               self.local_AC.apply_grads],
                                              feed_dict=feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def work(self, max_episode_length, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0
                d = False

                s = self.env.reset()
                episode_frames.append(s)
                s = process_pong(s)
                rnn_state = self.local_AC.lstm.state_init

                while d == False:
                    # Take an action using probabilities from policy network output.
                    a_dist, v, rnn_state = sess.run([self.local_AC.policy, self.local_AC.value, self.local_AC.lstm.state_out],
                                                    feed_dict={self.local_AC.conv.input_raw: [s],
                                                               self.local_AC.lstm.state_in[0]: rnn_state[0],
                                                               self.local_AC.lstm.state_in[1]: rnn_state[1]})
                    a = np.random.choice(a_dist[0], p=a_dist[0])
                    a = np.argmax(a_dist == a)

                    action = None  # self.actions[a]
                    if a == 0:
                        action = 0
                    elif a == 1:
                        action = 2
                    elif a == 2:
                        action = 3
                    else:
                        raise Exception('no action')

                    s1, r, d, _ = self.env.step(action)
                    r = r / 100.0
                    if d == False:
                        episode_frames.append(s1)
                        s1 = process_pong(s1)
                    else:
                        s1 = s

                    episode_buffer.append([s, a, r, s1, d, v[0, 0]])
                    episode_values.append(v[0, 0])

                    episode_reward += r
                    s = s1
                    total_steps += 1
                    episode_step_count += 1

                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if len(episode_buffer) == 30 and d != True and episode_step_count != max_episode_length - 1:
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        v1 = sess.run(self.local_AC.value,
                                      feed_dict={self.local_AC.conv.input_raw: [s],
                                                 self.local_AC.lstm.state_in[0]: rnn_state[0],
                                                 self.local_AC.lstm.state_in[1]: rnn_state[1]})[0, 0]
                        v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    if d == True:
                        break

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # Update the network using the experience buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    v_l, p_l, e_l, g_n, v_n = self.train(
                        episode_buffer, sess, gamma, 0.0)

                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    if self.wid == 'worker_0' and episode_count % 25 == 0:
                        time_per_step = 0.05
                        images = np.array(episode_frames)
                    if episode_count % 250 == 0 and self.wid == 'worker_0':
                        saver.save(sess, self.model_path +
                                   '/model-' + str(episode_count) + '.cptk')
                        print("Saved Model")

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
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
                if self.wid == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1


if __name__ == '__main__':
    max_episode_length = 300
    gamma = .99  # discount rate for advantage estimation and reward discounting
    state_size = 80 * 80 * 1  # Observations are greyscale frames of 80 * 80 * 1
    action_size = 3  # Agent can move Left, Right, or Stay
    load_model = False
    model_path = './model'
    xlen, ylen, chans = 80, 80, 1

    tf.reset_default_graph()

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    with tf.device("/cpu:0"):
        global_episodes = tf.Variable(
            0, dtype=tf.int32, name='global_episodes', trainable=False)
        trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
        # Generate global network
        master_network = ActorCriticNetwork(state_size, action_size, 'global', None, xlen, ylen, chans)
        # Set workers ot number of available CPU threads
        num_workers = multiprocessing.cpu_count()
        workers = []
        # Create worker classes
        for i in range(num_workers):
            workers.append(Worker(gym.make('Pong-v0'), i, state_size,
                                  action_size, trainer, model_path, global_episodes))
        saver = tf.train.Saver(max_to_keep=5)

    with tf.Session(config=sess_config) as sess:
        coord = tf.train.Coordinator()
        if load_model == True:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        # This is where the asynchronous magic happens.
        # Start the "work" process for each worker in a separate threat.
        worker_threads = []
        for worker in workers:
            def worker_work(): return worker.work(
                max_episode_length, gamma, sess, coord, saver)
            t = threading.Thread(target=(worker_work))
            t.start()
            sleep(0.5)
            worker_threads.append(t)
        coord.join(worker_threads)
