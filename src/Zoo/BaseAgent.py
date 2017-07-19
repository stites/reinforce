from Zoo.Prelude import *

class BaseAgent(object):
    def __init__(self, load_model, pretrain_steps, path):
        self.load_model = load_model
        self.pretrain_steps = pretrain_steps
        self.path = path

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
        return not (self.pretrain_steps == None) or steps > self.pretrain_steps


