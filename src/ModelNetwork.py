from Zoo.Prelude import *

class Agent:
    def __init__(
            self,
            env,
            max_episodes=2000,
            max_steps=299,
            batch_size=5,
            batch_size_model=5,
            hidden_size=8,
            hidden_size_model=256,
            lr=1e-2,
            gamma=0.99,
            train_on_model_only=True
            ):
        _, self.state_size       = space_sizes(env)
        self.action_size         = 1
        self.train_on_model_only = train_on_model_only

        self.env               = env
        self.max_episodes      = max_episodes
        self.max_steps         = max_steps
        self.batch_size        = batch_size
        self.batch_size_model  = batch_size_model
        self.hidden_size       = hidden_size
        self.hidden_size_model = hidden_size_model
        self.lr                = lr
        self.gamma             = gamma

        """ Build the tensorflow graph """
        tf.reset_default_graph()
        self.policy = self.Policy(self.state_size, self.action_size, hidden_size, lr, gamma)
        self.model  = self.Model(self.state_size, self.action_size, hidden_size_model, lr)


    def run_learner(self):
        """ run the model against the environment """
        policy, model, env = self.policy, self.model, self.env
        resetGradBuffer = lambda gbuffer: lmap(mul(0), gbuffer)

        running_reward = 0
        reward_sum = 0
        real_episodes = 1
        init = tf.global_variables_initializer()

        self.using_model  = False
        self.train_policy = False

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            gradBuffer = resetGradBuffer(sess.run(policy.tvars))

            for episode_number in range(self.max_episodes):
                done = False
                step_num = 0
                xs,drs,ys,ds = [],[],[],[] # reset array memory
                x, active_batch_size = self.reset(self.using_model, env)

                while not done and step_num < self.max_steps:
                    step_num += 1
                    action = policy.epsilon_greedy_choice(sess, x)

                    # record various intermediates (needed later for backprop)
                    y = 1 if action == 0 else 0
                    next_x, reward, done = self.step(self.using_model, sess, x, action, model, env)
                    reward_sum += reward

                    ys.append(y)
                    xs.append(x)
                    ds.append(done*1)
                    drs.append(reward)
#                    history.tell(x, reward, action, int(done))

                    x = next_x
                """ episode completed """

                real_episodes += int(not self.using_model)

                epx = np.vstack(xs)
                epy = np.vstack(ys)
                epr = np.vstack(drs)
                epd = np.vstack(ds)

                if not self.using_model:
                    """ we always train the model when it is not in use """
                    loss, pState, all_nexts = model.train(sess, epx, epy, epr, epd)

                if self.train_policy:
                    """ we may choose to train the policy only when the model is in use """
                    nGrads = policy.find_gradients(sess, epx, epy, epr)
                    gradBuffer = lmap(curry(lambda a, b: a+b), zip(nGrads, gradBuffer))

                if (episode_number - 1) % active_batch_size == 0 and episode_number != 1:
                    if self.train_policy:
                        policy.train(sess, gradBuffer)
                        gradBuffer = resetGradBuffer(gradBuffer)

                    running_reward = truncate(running_reward) * 0.99 + truncate(reward_sum) * 0.01

                    if not self.using_model:
                        info = (real_episodes, reward_sum/self.batch_size, action, truncate(running_reward/self.batch_size))
                        print('World Perf: Episode %f. Reward %f. action: %f. mean reward %f.' % info)

                    reward_sum = 0

                self.using_model  = self.toggle(episode_number, active_batch_size, self.using_model)
                self.train_policy = self.using_model if self.train_on_model_only else True

        print('%i episodes run from gym environment, %i simulated episodes' % (real_episodes, self.max_episodes - real_episodes))
        return pState, all_nexts


    def toggle(self, episode_number, active_batch_size, flag):
        """
        Once the model has been trained on 100 episodes, we start alternating between training the
        policy from the model and training the model from the real environment.
        """
        if ((episode_number - 1) % active_batch_size == 0) and (episode_number > 100):
            return not flag
        else:
            return flag


    def step(self, drawFromModel, sess, state, action, model, env):
        """ step the model or real environment and get new measurements """
        if drawFromModel:
            obs, rwd, done = model.step(sess, state, action)
        else:
            obs, rwd, done, _ = env.step(action)
            obs = obs[np.newaxis]

        return obs, rwd, done


    def reset(self, drawFromModel, env):
        if drawFromModel:
            observation = np.random.uniform(-0.1,0.1,[1,4]) # random default starting point
            batch_size  = self.batch_size_model
        else:
            observation = np.array([env.reset()])
            batch_size  = self.batch_size
        return observation, batch_size


    class Policy:
        def __init__(self, ssize, asize, hsize, lr, gamma):
            self.gamma = gamma

            with tf.variable_scope('policy') as scope:
                inputs = tf.placeholder(shape=[None, ssize], dtype=tf.float32, name="input_x")

                w1     = tf.get_variable("w1", shape=[ssize, hsize], initializer=tf.contrib.layers.xavier_initializer())
                layer1 = tf.nn.relu(tf.matmul(inputs, w1))
                w2     = tf.get_variable("w2", shape=[hsize, 1], initializer=tf.contrib.layers.xavier_initializer())
                output = tf.matmul(layer1, w2)

                probability = tf.nn.sigmoid(output)

                tvars         = [v for v in tf.trainable_variables() if v.name.startswith(scope.name)]
                input_acts = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="input_acts")
                advantages    = tf.placeholder(dtype=tf.float32, name="reward_signal")
                adam          = tf.train.AdamOptimizer(learning_rate=lr)

                w1Grad    = tf.placeholder(dtype=tf.float32, name="batch_grad1")
                w2Grad    = tf.placeholder(dtype=tf.float32, name="batch_grad2")
                batchGrad = [w1Grad, w2Grad]

                llikelihood = tf.log(input_acts * (input_acts - probability) + (1 - input_acts) * (input_acts + probability))
                loss        = -tf.reduce_mean(llikelihood * advantages)
                newGrads    = tf.gradients(loss, tvars)
                updateGrads = adam.apply_gradients(zip(batchGrad, tvars))

            self.tvars = tvars
            self.score = output
            self.probability = probability
            self.observations = inputs
            self.advantages = advantages
            self.updateGrads = updateGrads
            self.input_acts = input_acts
            self.newGrads = newGrads
            self.w1Grad = w1Grad
            self.w2Grad = w2Grad


        def epsilon_greedy_choice(self, sess, x):
            model_prob = sess.run(self.probability, feed_dict={self.observations: x})
            return 1 if np.random.uniform() < model_prob else 0


        def find_gradients(self, sess, epx, epy, epr):
            discounted_epr  = eligibility_trace(self.gamma, epr)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            nGrads = sess.run(self.newGrads, feed_dict={
                self.observations: epx,
                self.input_acts: epy,
                self.advantages: discounted_epr
            })

            if np.sum(nGrads[0] == nGrads[0]) == 0:
                msg = "the gradients have become too large, this may be okay but I recommend you doublecheck that this is expected"
                raise Exception(msg)

            return nGrads


        def train(self, sess, gradBuffer):
            sess.run(self.updateGrads, feed_dict={
                self.w1Grad:gradBuffer[0],
                self.w2Grad:gradBuffer[1]
            })


    class Model:
        def __init__(self, ssize, asize, hsize, lr):
            xavier = tf.contrib.layers.xavier_initializer

            with tf.variable_scope('model'):
                # inputs are the current state and action:
                previous_state = tf.placeholder(tf.float32, [None, ssize+asize] , name="previous_state")

                # start our first hidden layer of our model's neural network
                w1     = tf.get_variable(name="w1", shape=[ssize+asize, hsize], initializer=xavier())
                b1     = tf.Variable(tf.zeros([hsize]), name="b1")
                layer1 = tf.nn.relu(tf.matmul(previous_state, w1) + b1)

                # and again with our second hidden layer, which is locked in at the same size as our first layer.
                w2     = tf.get_variable(name="w2", shape=[hsize, hsize], initializer=xavier())
                b2     = tf.Variable(tf.zeros([hsize]), name="b2")
                layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)

                # Next we map different outputs depending on what we want out of our model. Each output layer has a
                # tensor of weights and biases, and we must also store a tensor of true values to approximate. The
                # output layers includes a layer fo state, reward, and completion.


                # An output layer for the observed state
                w_observation         = tf.get_variable("w_observation", shape=[hsize, ssize], initializer=xavier())
                b_observation         = tf.Variable(tf.zeros([4]), name="b_observation")
                predicted_observation = tf.matmul(layer2, w_observation, name="predicted_observation") + b_observation
                true_observation      = tf.placeholder(tf.float32, shape=[None, ssize], name="true_observation")
                # talk a bit more about this loss function
                observation_loss      = tf.square(true_observation - predicted_observation)

                # An estimated reward
                w_reward         = tf.get_variable("w_reward", shape=[hsize, 1], initializer=xavier())
                b_reward         = tf.Variable(tf.zeros([1]), name="b_reward")
                predicted_reward = tf.matmul(layer2, w_reward, name="predicted_reward") + b_reward
                true_reward      = tf.placeholder(tf.float32, shape=[None, 1], name="true_reward")
                # same loss as with observation
                reward_loss      = tf.square(true_reward - predicted_reward)

                # and a boolean of whether or not the episode has completed.
                w_done         = tf.get_variable("w_done", shape=[hsize, 1], initializer=xavier())
                b_done         = tf.Variable(tf.ones([1]), name="b_done")
                predicted_done = tf.sigmoid(tf.matmul(layer2, w_done, name="predicted_done") + b_done)
                true_done      = tf.placeholder(tf.float32, shape=[None, 1], name="true_done")
                # talk a bit more about this loss function
                done_loss      = -tf.log(tf.multiply(predicted_done, true_done) + tf.multiply(1-predicted_done, 1-true_done))

                # the final output is actually simple the concatenation of all predicted outcomes
                predicted_state = tf.concat([predicted_observation, predicted_reward, predicted_done], 1)

                # and we want to minimize the aggregate loss across all predicted outcomes
                model_loss   = tf.reduce_mean(observation_loss + done_loss + reward_loss)
                update_model = tf.train.AdamOptimizer(learning_rate=lr).minimize(model_loss)

            self.previous_state   = previous_state
            self.predicted_state  = predicted_state
            self.model_loss       = model_loss
            self.update_model     = update_model
            self.true_observation = true_observation
            self.true_done        = true_done
            self.true_reward      = true_reward


        def step(self, sess, xs, action):
            """ This function uses our model to produce a new state when given a previous state and action """
            toFeed           = np.reshape(np.hstack([xs[-1], np.array(action)]),[1,5])
            myPredict        = sess.run([self.predicted_state], feed_dict = {self.previous_state: toFeed})
            reward           = float(myPredict[0][:,4])
            observation      = myPredict[0][:,0:4]
            observation[:,0] = np.clip(observation[:,0],-2.4,2.4)
            observation[:,2] = np.clip(observation[:,2],-0.4,0.4)
            done             = np.clip(myPredict[0][:,5],0,1) > 0.1 or len(xs) >= 300
            return observation, reward, done


        def train(self, sess, epx, epy, epr, epd):
            """
            here we use the tail for all historical inputs that contain true information and use the last n-1
            states to predict where we will end up next.
            """
            actions = np.array([np.abs(y-1) for y in epy])
            state_prevs = np.hstack([epx, actions])[ :-1,:]
            state_nexts =                       epx[1:  ,:]
            rewards     =              np.array(epr[1:  ,:])
            dones       =              np.array(epd[1:  ,:])
            all_nexts = np.hstack([state_nexts,rewards,dones])

            loss, pState, _ = sess.run([
                self.model_loss,
                self.predicted_state,
                self.update_model
            ], feed_dict={
                self.previous_state  : state_prevs,
                self.true_observation: state_nexts,
                self.true_reward     : rewards,
                self.true_done       : dones
            })

            return loss, pState, all_nexts


if __name__ == "__main__":
    agent = Agent(gym.make('CartPole-v0'), max_episodes=2000, max_steps=250)
    agent.run_learner()




