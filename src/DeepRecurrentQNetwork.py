from Zoo.Prelude       import *
from Zoo.Gridworld     import Gridworld
from Zoo.BasicConvoNet import ConvolutionNetwork
from Zoo.ReplayBuffer  import ReplayBuffer

class DuelingRecurrentNetwork():
    class RecurrentLayer:
        def __init__(self, output, hidden_size, rnn_cell):
            """
            The input must be reshaped into [batch x trace x units] for rnn processing,
            and then returned to [batch x units] when sent through the upper levels.
            """
            batch_size   = tf.placeholder(dtype=tf.int32)
            train_length = tf.placeholder(dtype=tf.int32)
            conv_flat    = tf.reshape(slim.flatten(output), [batch_size, train_length, hidden_size])

            state_in = rnn_cell.zero_state(batch_size, tf.float32)

            rnn, rnn_state = tf.nn.dynamic_rnn(
                    inputs=conv_flat,
                    cell=rnn_cell,
                    dtype=tf.float32,
                    initial_state=state_in)

            self.output       = tf.reshape(rnn, shape=[-1, hidden_size])
            self.batch_size   = batch_size
            self.train_length = train_length

    def __init__(self, xlen, ylen, chans, hidden_size, action_size, rnn_cell, scope_name):
        cnn         = ConvolutionNetwork(84, 84, 3, h_size)
        self.rnn = self.RecurrentLayer(cnn.output, hidden_size, rnn_cell)

        # Take the output from the convolutional layer and send it to a recurrent layer.

        #The output from the recurrent player is then split into separate Value and Advantage streams
        advantageStream, valueStream = tf.split(rnn.output, 2, 1) ## find out what these parameters are doing!

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
        shape_half  = [batch_size, int(train_length/2)]
        mask        = tf.reshape(tf.concat([tf.zeros(shape_half), tf.ones(shape_half)], 1), [-1])
        loss        = tf.reduce_mean(td_error * mask)
        updateModel = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

        # """ things we need """
        self.scalarInput = cnn.scalar_input
        # self.qOut        = qOut
        # self.predict     = predict
        # self.targetQ     = targetQ
        # self.actions     = actions
        # self.updateModel = updateModel



#This is a simple function to reshape our game frames.
def processState(state1):
    return np.reshape(state1,[21168])

#These functions allows us to update the parameters of our target network with those of the primary network.
def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)
    total_vars = len(tf.trainable_variables())
    a = tf.trainable_variables()[0].eval(session=sess)
    b = tf.trainable_variables()[total_vars//2].eval(session=sess)
    if not a.all() == b.all():
        raise Exception("Target Set Failed")


if __name__ == '__main__':
    env = Gridworld(partial=False,sizeX=9)
    print(env)
    #Setting the training parameters
    batch_size = 4 #How many experience traces to use for each training step.
    trace_length = 8 #How long each experience trace will be when training
    update_freq = 5 #How often to perform a training step.
    y = .99 #Discount factor on the target Q-values
    startE = 1 #Starting chance of random action
    endE = 0.1 #Final chance of random action
    anneling_steps = 10000 #How many steps of training to reduce startE to endE.
    num_episodes = 10000 #How many episodes of game environment to train network with.
    pre_train_steps = 10000 #How many steps of random actions before training begins.
    load_model = False #Whether to load a saved model.
    path = "./drqn" #The path to save our model to.
    h_size = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
    max_epLength = 50 #The max allowed length of our episode.
    time_per_step = 1 #Length of each step used in gif creation
    summaryLength = 100 #Number of epidoes to periodically save for analysis
    tau = 0.001

    tf.reset_default_graph()
    #We define the cells for the primary and target q-networks
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=h_size,state_is_tuple=True)
    cellT = tf.contrib.rnn.BasicLSTMCell(num_units=h_size,state_is_tuple=True)
    mainQN = DuelingRecurrentNetwork(h_size,cell,'main')
    targetQN = DuelingRecurrentNetwork(h_size,cellT,'target')

    init = tf.global_variables_initializer()

    saver = tf.train.Saver(max_to_keep=5)

    trainables = tf.trainable_variables()

    targetOps = updateTargetGraph(trainables,tau)

    myBuffer = ReplayBuffer(buffer_size = 1000)

    #Set the rate of random action decrease. 
    e = startE
    stepDrop = (startE - endE)/anneling_steps

    #create lists to contain total rewards and steps per episode
    jList = []
    rList = []
    total_steps = 0

    #Make a path for our model to be saved in.
    if not os.path.exists(path):
        os.makedirs(path)

    ##Write the first line of the master log-file for the Control Center
    with open('./Center/log.csv', 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(['Episode','Length','Reward','IMG','LOG','SAL'])


    with tf.Session() as sess:
        if load_model == True:
            print ('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(path)
            saver.restore(sess,ckpt.model_checkpoint_path)
        sess.run(init)

        updateTarget(targetOps,sess) #Set the target network to be equal to the primary network.
        for i in range(num_episodes):
            episodeBuffer = []
            #Reset environment and get first new observation
            sP = env.reset()
            s = processState(sP)
            d = False
            rAll = 0
            j = 0
            state = (np.zeros([1,h_size]),np.zeros([1,h_size])) #Reset the recurrent layer's hidden state
            #The Q-Network
            while j < max_epLength: 
                j+=1
                #Choose an action by greedily (with e chance of random action) from the Q-network
                if np.random.rand(1) < e or total_steps < pre_train_steps:
                    state1 = sess.run(mainQN.rnn_state,\
                        feed_dict={mainQN.scalarInput:[s/255.0],mainQN.trainLength:1,mainQN.state_in:state,mainQN.batch_size:1})
                    a = np.random.randint(0,4)
                else:
                    a, state1 = sess.run([mainQN.predict,mainQN.rnn_state],\
                        feed_dict={mainQN.scalarInput:[s/255.0],mainQN.trainLength:1,mainQN.state_in:state,mainQN.batch_size:1})
                    a = a[0]
                s1P,r,d = env.step(a)
                s1 = processState(s1P)
                total_steps += 1
                episodeBuffer.append(np.reshape(np.array([s,a,r,s1,d]),[1,5]))
                if total_steps > pre_train_steps:
                    if e > endE:
                        e -= stepDrop

                    if total_steps % (update_freq) == 0:
                        updateTarget(targetOps,sess)
                        #Reset the recurrent layer's hidden state
                        state_train = (np.zeros([batch_size,h_size]),np.zeros([batch_size,h_size])) 

                        trainBatch = myBuffer.sample_sequence(batch_size, trace_length) #Get a random batch of experiences.
                        #Below we perform the Double-DQN update to the target Q-values
                        Q1 = sess.run(mainQN.predict,feed_dict={\
                            mainQN.scalarInput:np.vstack(trainBatch[:,3]/255.0),\
                            mainQN.trainLength:trace_length,mainQN.state_in:state_train,mainQN.batch_size:batch_size})
                        Q2 = sess.run(targetQN.Qout,feed_dict={\
                            targetQN.scalarInput:np.vstack(trainBatch[:,3]/255.0),\
                            targetQN.trainLength:trace_length,targetQN.state_in:state_train,targetQN.batch_size:batch_size})
                        end_multiplier = -(trainBatch[:,4] - 1)
                        doubleQ = Q2[range(batch_size*trace_length),Q1]
                        targetQ = trainBatch[:,2] + (y*doubleQ * end_multiplier)
                        #Update the network with our target values.
                        sess.run(mainQN.updateModel, \
                            feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,0]/255.0),mainQN.targetQ:targetQ,\
                            mainQN.actions:trainBatch[:,1],mainQN.trainLength:trace_length,\
                            mainQN.state_in:state_train,mainQN.batch_size:batch_size})
                rAll += r
                s = s1
                sP = s1P
                state = state1
                if d == True:
                    break

            #Add the episode to the experience buffer
            bufferArray = np.array(episodeBuffer)
            episodeBuffer = list(zip(bufferArray))
            myBuffer.append(episodeBuffer)
            jList.append(j)
            rList.append(rAll)

            #Periodically save the model. 
            if i % 1000 == 0 and i != 0:
                saver.save(sess,path+'/model-'+str(i)+'.cptk')
                print ("Saved Model")
            if len(rList) % summaryLength == 0 and len(rList) != 0:
                print (total_steps,np.mean(rList[-summaryLength:]), e)
                saveToCenter(i,rList,jList,np.reshape(np.array(episodeBuffer),[len(episodeBuffer),5]),\
                    summaryLength,h_size,sess,mainQN,time_per_step)
        saver.save(sess,path+'/model-'+str(i)+'.cptk')
    e = 0.01 #The chance of chosing a random action
    num_episodes = 10000 #How many episodes of game environment to train network with.
    load_model = True #Whether to load a saved model.
    path = "./drqn" #The path to save/load our model to/from.
    h_size = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
    h_size = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
    max_epLength = 50 #The max allowed length of our episode.
    time_per_step = 1 #Length of each step used in gif creation
    summaryLength = 100 #Number of epidoes to periodically save for analysis

    tf.reset_default_graph()
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=h_size,state_is_tuple=True)
    cellT = tf.contrib.rnn.BasicLSTMCell(num_units=h_size,state_is_tuple=True)
    mainQN = DuelingRecurrentNetwork(h_size,cell,'main')
    targetQN = DuelingRecurrentNetwork(h_size,cellT,'target')

    init = tf.global_variables_initializer()

    saver = tf.train.Saver(max_to_keep=2)

    #create lists to contain total rewards and steps per episode
    jList = []
    rList = []
    total_steps = 0

    #Make a path for our model to be saved in.
    if not os.path.exists(path):
        os.makedirs(path)

    ##Write the first line of the master log-file for the Control Center
    with open('./Center/log.csv', 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(['Episode','Length','Reward','IMG','LOG','SAL'])

        #wr = csv.writer(open('./Center/log.csv', 'a'), quoting=csv.QUOTE_ALL)
    with tf.Session() as sess:
        if load_model == True:
            print ('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(path)
            saver.restore(sess,ckpt.model_checkpoint_path)
        else:
            sess.run(init)


        for i in range(num_episodes):
            episodeBuffer = []
            #Reset environment and get first new observation
            sP = env.reset()
            s = processState(sP)
            d = False
            rAll = 0
            j = 0
            state = (np.zeros([1,h_size]),np.zeros([1,h_size]))
            #The Q-Network
            while j < max_epLength: #If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
                j+=1
                #Choose an action by greedily (with e chance of random action) from the Q-network
                if np.random.rand(1) < e:
                    state1 = sess.run(mainQN.rnn_state,\
                        feed_dict={mainQN.scalarInput:[s/255.0],mainQN.trainLength:1,mainQN.state_in:state,mainQN.batch_size:1})
                    a = np.random.randint(0,4)
                else:
                    a, state1 = sess.run([mainQN.predict,mainQN.rnn_state],\
                        feed_dict={mainQN.scalarInput:[s/255.0],mainQN.trainLength:1,\
                        mainQN.state_in:state,mainQN.batch_size:1})
                    a = a[0]
                s1P,r,d = env.step(a)
                s1 = processState(s1P)
                total_steps += 1
                episodeBuffer.append(np.reshape(np.array([s,a,r,s1,d]),[1,5])) #Save the experience to our episode buffer.
                rAll += r
                s = s1
                sP = s1P
                state = state1
                if d == True:

                    break

            bufferArray = np.array(episodeBuffer)
            jList.append(j)
            rList.append(rAll)

            #Periodically save the model. 
            if len(rList) % summaryLength == 0 and len(rList) != 0:
                print (total_steps,np.mean(rList[-summaryLength:]), e)
                saveToCenter(i,rList,jList,np.reshape(np.array(episodeBuffer),[len(episodeBuffer),5]),\
                    summaryLength,h_size,sess,mainQN,time_per_step)
        print ("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")

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

