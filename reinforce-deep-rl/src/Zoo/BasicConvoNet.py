from Zoo.Prelude import *

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
        self.imageIn      = imageIn


