from Zoo.Prelude      import *
from Zoo.ReplayBuffer import ReplayBuffer
from Zoo.TensorOps    import update_target_graph
from Zoo.BaseAgent    import BaseAgent

# weight initializers
def truncated_normal_initializer():
    return tf.truncated_normal_initializer(mean=0.0, stddev=0.02, dtype=tf.float32)


def uniform_initializer():
    return tf.random_uniform_initializer(minval=-0.003, maxval=0.003, dtype=tf.float32)

# layer definitions
class Layer(object):
    """layer of a neural network"""
    def __init__(self,
            inputs_:Union[Any, int],
            out_size:int,
            weight_initialization:Callable[[], Any],
            activation:Callable[[Any], Any]
        )->None:
        super(Layer, self).__init__()
        batch_size:int = None  # for posterity
        in_size:int    = None
        if isinstance(inputs_, tf.Tensor):
            in_size = inputs_.get_shape().as_list()[1]  # assume this is a 2d tensor
            inputs  = inputs_
        else:
            in_size = inputs_
            inputs  = tf.placeholder(shape=[None, in_size], dtype=tf.float32)

        inputs_flat = inputs # tf.reshape(inputs, [-1, batch_size])
        weights     = tf.Variable(weight_initialization()([in_size, out_size]), dtype=tf.float32)
        bias        = tf.Variable(tf.zeros([out_size]), dtype=tf.float32)
        outputs     = activation(tf.matmul(inputs_flat, weights) + bias)

        self.inputs, self.bias, self.outputs, self.weights = inputs, bias, outputs, weights


class TwoInputLayer(object):
    """ critic layer 2 pulls in the actions as an input """
    def __init__(self,
            input1:Any,
            input2:Any,
            out_size:int,
            weight_initialization:Callable[[], Any],
            activation:Callable[[Any], Any]
        )->None:
        super(TwoInputLayer, self).__init__()
        """ not going to try to modify this to taking an input size at all """

        in1_size = input1.get_shape().as_list()[-1]
        in2_size = input2.get_shape().as_list()[-1]

        with tf.name_scope("TwoInputLayer"):
            weights1 = tf.Variable(weight_initialization()([in1_size, out_size]))
            weights2 = tf.Variable(weight_initialization()([in2_size, out_size]))
            bias     = tf.Variable(tf.zeros([out_size]))
            outputs  = activation(tf.matmul(input1, weights1) + tf.matmul(input2, weights2) + bias)

        self.input1  = input1
        self.input2  = input2
        self.outputs = outputs


class ActorNetwork(object):
    """ build out a deterministic policy network """
    def __init__(self, state_space:int, action_space:int, action_boundary:int, learning_rate:float, is_main:bool=False)->None:
        super(ActorNetwork, self).__init__()
        with tf.name_scope('actor_net_'+('main' if is_main else 'target')) as scope:
            self.scope = scope
            self.inputs, self.outputs = self.build(state_space, action_space)
            self.outputs_scaled       = tf.multiply(self.outputs, action_boundary)

            self.optimize_op, self.action_gradient = \
                    self.build_optimize_op(self.outputs, action_space, scope, learning_rate)

    def build(self, state_space:int, action_space:int):
        layer1 = Layer(   state_space,          400, truncated_normal_initializer, tf.nn.relu)
        layer2 = Layer(layer1.outputs,          300, truncated_normal_initializer, tf.nn.relu)
        layer3 = Layer(layer2.outputs, action_space,          uniform_initializer,    tf.tanh)
        return layer1.inputs, layer3.outputs

    def build_optimize_op(self, outputs, out_size, scope_name, learning_rate):
        train_gradient  = tf.placeholder(tf.float32, [None, out_size])
        network_params  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_name)
        gradients       = tf.gradients(self.outputs, network_params, -train_gradient)
        trainer         = tf.train.AdamOptimizer(learning_rate)
        return trainer.apply_gradients(zip(gradients, network_params)), train_gradient

    def predict(self, sess, states):
        return sess.run(self.outputs_scaled, feed_dict={
            self.inputs: states
        })

class CriticNetwork(object):
    """ build out a compatible function approximator """
    def __init__(self, state_space, action_space, learning_rate, is_main=False):
        super(CriticNetwork, self).__init__()
        with tf.name_scope('critic_net_'+('stable' if is_main else 'online')) as scope:
            self.scope = scope
            self.states_input, self.actions_input, self.outputs = \
                self.build(state_space, action_space)

            # these gradients will be passed back to the actor network
            self.action_gradients = tf.gradients(self.outputs, self.actions_input)
            self.optimize_op, self.predicted_qs = \
                    self.build_optimize_op(self.outputs, learning_rate)

    def build(self, state_space:int, action_space:int):
        states  = tf.placeholder(dtype=tf.float32, shape=[None,  state_space])
        actions = tf.placeholder(dtype=tf.float32, shape=[None, action_space])
        layer1 = Layer(states, 400, truncated_normal_initializer, tf.nn.relu)
        layer2 = TwoInputLayer(actions, layer1.outputs, 300, truncated_normal_initializer, tf.nn.relu)
        layer3 = Layer(layer2.outputs, 1, uniform_initializer, identity1)

        return states, actions, layer3.outputs

    def build_optimize_op(self, outputs, learning_rate):
        predicted_qs = tf.placeholder(tf.float32, [None, 1])
        loss         = tf.reduce_mean(tf.square(predicted_qs - outputs))
        trainer      = tf.train.AdamOptimizer(learning_rate)
        return trainer.minimize(loss), predicted_qs

    def predict(self, sess, states, actions):
        return sess.run(self.outputs, feed_dict={
            self.states_input: states,
            self.actions_input: actions
        })



class Agent(BaseAgent):
    """Actor-Critic agent"""
    def __init__(self, ssize, asize, aboundary, actor_lr, critic_lr, tau, env, gamma=0.99, load_model=False, pretrain_steps=None):
        super(Agent, self).__init__(load_model, pretrain_steps, path="./ddpg/", env=env)
        self.tau = tau
        self.env = env
        self.gamma = gamma
        self.asize = asize
        self.ssize = ssize

        self.behaviour_policy = ActorNetwork(ssize, asize, aboundary, actor_lr, is_main=True)
        self.target_policy    = ActorNetwork(ssize, asize, aboundary, actor_lr, is_main=False)
        self.sync_target_policy_op = update_target_graph(self.behaviour_policy.scope, self.target_policy.scope, self.tau)

        self.stable_value_function = CriticNetwork(ssize, asize, critic_lr, is_main=True)
        self.online_value_function = CriticNetwork(ssize, asize, critic_lr, is_main=False)
        self.sync_target_value_op = update_target_graph(self.stable_value_function.scope, self.online_value_function.scope, self.tau)

    def train_behaviour_policy(self, sess, states, critic_action_gradients)->None:
        policy = self.behaviour_policy
        sess.run(policy.optimize_op, feed_dict={
            policy.inputs: states,
            policy.action_gradient: critic_action_gradients
        })

    def train_stable_value(self, sess, states, actions, est_values)->Tuple[Any, Any]:
        value = self.stable_value_function
        return sess.run([value.outputs, value.optimize_op], feed_dict={
            value.states_input: states,
            value.actions_input: actions,
            value.predicted_qs: est_values
        })

    def stable_action_gradients(self, sess, states, actions)->Any:
        value = self.stable_value_function
        return sess.run(value.action_gradients, feed_dict={
            value.states_input: states,
            value.actions_input: actions
        })

    def process_state(self, state):
        return np.reshape(state, (1, self.ssize))

    def run_learner(self, buffer_size=1000000, batch_size=64, max_episodes=50000, max_steps=1000):
        saver = tf.train.Saver()
        experience       = ReplayBuffer(buffer_size)
        behaviour_policy = self.behaviour_policy
        target_policy    = self.target_policy
        online_critic    = self.online_value_function
        stable_critic    = self.stable_value_function

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter('./.tensorboard/w1', sess.graph)

            self.load(sess, saver)
            sess.run(self.sync_target_policy_op)
            sess.run(self.sync_target_value_op)


            total_steps = 0
            _all_rewards = []

            for ep_num in range(max_episodes):
                state    = self.reset()
                done     = False
                step_num = 0

                _epRwds  = 0.
                _epMaxQs = 0.
                while step_num < max_steps and not done:
                    step_num += 1
                    total_steps += 1

                    ornstien_uhlenbeck_momentum_noise = 1. / (1. + ep_num)  # from ddpg paper
                    action = behaviour_policy.predict(sess, state)[0] + ornstien_uhlenbeck_momentum_noise
                    next_state, reward, done, _ = self.step(action)
                    experience.add_step(state, action, reward, next_state, done)
                    state = next_state

                    _epRwds += float(reward)

                    if self.finished_pretrain(total_steps):
                        ss, _as, rs, s_nexts, ds = experience.sample_batch_split(batch_size)
                        target_qs = online_critic.predict(sess, s_nexts, target_policy.predict(sess, s_nexts))

                        def discount(is_terminal, reward, target):
                            return reward + (0 if is_terminal else self.gamma * target)

                        td_error = lmap(lambda3(discount), zip(ds, rs, target_qs))

                        predicted_qs, _  = self.train_stable_value(sess, ss, _as, td_error)
                        _epMaxQs        += np.amax(predicted_qs)
                        action_policies  = behaviour_policy.predict(sess, ss)
                        action_gradients = self.stable_action_gradients(sess, ss, action_policies)[0]
                        self.train_behaviour_policy(sess, ss, action_gradients)

                        sess.run(self.sync_target_policy_op)
                        sess.run(self.sync_target_value_op)


                    if done:
                        _all_rewards.append(_epRwds)
                        last100 = _all_rewards[-100:]
                        averageQMax = _epMaxQs / float(step_num)
                        summary = tf.Summary()
                        summary.value.add(tag='Episode/Reward', simple_value=_epRwds)
                        summary.value.add(tag='Episode/Average_QMax', simple_value=averageQMax)
                        summary.value.add(tag='Episodes/Rolling100Reward', simple_value=sum(last100)/float(len(last100)))
                        summary.value.add(tag='Episodes/Length', simple_value=step_num)
                        writer.add_summary(summary, ep_num)
                        writer.flush()

                        print('| Reward: %.2i' % int(_epRwds), " | Episode", ep_num, '| Qmax: %.4f' % averageQMax)


if __name__ == '__main__':
    env          = gym.make('Pendulum-v0')
    state_size   = env.observation_space.shape[0]
    action_size  = env.action_space.shape[0]
    action_bound = env.action_space.high

    Agent(
        ssize=state_size,
        asize=action_size,
        aboundary=action_bound,
        actor_lr=0.0001,
        critic_lr=0.001,
        tau=0.001,
        env=env,
        load_model=False,
        gamma=0.9,
        pretrain_steps=64) \
     .run_learner(buffer_size=1000000, batch_size=64, max_episodes=50000, max_steps=1000)

