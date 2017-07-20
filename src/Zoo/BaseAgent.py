from Zoo.Prelude import *

class BaseAgent(object):
    """ basic helper functions which should live on an agent """
    def __init__(self, load_model, pretrain_steps, path, env):
        self.load_model = load_model
        self.pretrain_steps = 1 if pretrain_steps == None else pretrain_steps
        self.path = path
        self.env = env
        self.summary_writer = tf.summary.FileWriter("./.tensorboard/"+"w0/")

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

    def tb_report(self, episode_rewards, episode_lengths, episode_mean_values, episode_count):
        summary = tf.Summary()

        summary.value.add(tag='Episode/Reward', simple_value=float(np.mean(episode_rewards[-5:])))
        summary.value.add(tag='Episode/Length', simple_value=float(np.mean(episode_lengths[-5:])))
        summary.value.add(tag='Episode/Value', simple_value=float(np.mean(episode_mean_values[-5:])))

        #summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
        #summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
        #summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
        #summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
        #summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))

        self.summary_writer.add_summary(summary, episode_count)
        self.summary_writer.flush()

