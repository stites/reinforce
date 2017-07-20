from Zoo.Prelude import *

class BaseAgent(object):
    """ basic helper functions which should live on an agent """
    def __init__(self, load_model, pretrain_steps, path, env):
        self.load_model = load_model
        self.pretrain_steps = 1 if pretrain_steps == None else pretrain_steps
        self.path = path
        self.env = env

    def process_state(self, states):
        return states

    def load(self, sess, saver):
        """ load the model to a session from a saver """
        if self.load_model == True:
            print('Loading Model..')
            ckpt = tf.train.get_checkpoint_state("./tmp"+self.path)
            saver.restore(sess, ckpt.model_checkpoint_path)

    def save(self, sess, saver, i):
        saver.save(sess, "./tmp"+self.path+'/model-'+str(i)+'.cptk')
        print("Saved Model")

    def finished_pretrain(self, step):
        return step > self.pretrain_steps

    def step(self, action:int)->Tuple[Any, float, bool, Any]:
         state, rwd, done, info = self.env.step(action)
         return self.process_state(state), float(rwd), bool(done), info

    def reset(self)->Any:
         return self.process_state(self.env.reset())

