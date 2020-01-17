from keras.layers import Layer
from keras.layers import Lambda
import keras.backend as K
class Failout(Layer):
    """Applies Failout to the input.
    Failout consists in randomly dropping out entire output of a layer by setting output to 0 at each update during training time,
    which helps increase resilencey of a distributed neural network 
    # Arguments
        reliability: float between 0 and 1. Probability of node failure.
        seed: A Python integer to use as random seed.
    """
    def __init__(self, reliability, seed=None, **kwargs):
        super(Failout, self).__init__(**kwargs)
        self.seed = seed
        self.reliability = K.variable(reliability)
        self.has_failed = None

    def call(self, inputs, training=None):
        rand = K.random_uniform(K.variable(0).shape, seed = self.seed)
        # assumes that there is only one input in inputs
        fail = Lambda(lambda x: x * 0)
        self.has_failed = K.greater(rand, self.reliability)
        failed_inputs = K.switch(self.has_failed,fail(inputs),inputs)
        failout = K.in_train_phase(failed_inputs, inputs, training)
        return failout

    def get_config(self):
        config = {
                  'seed': self.seed,
                  'reliability': self.reliability,
                  'has_failed': self.has_failed
                }
        base_config = super(Failout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class InputMux(Layer):
    """
    Input Multiplexer for a node that receives input from more than one downstream nodes
    # Arguments
        node_has_failed: Boolean Tensor showing if a node has failer
    """
    def __init__(self, has_failed, input1, input2, name, **kwargs):
        super(InputMux, self).__init__(**kwargs)
        self.has_failed = has_failed
        self.input1 = input1
        self.input2 = input2
        self.name = name

    def call(self, inputs, training=None):
        input = K.switch(self.has_failed, self.input1, self.input2)
        mux = Lambda(lambda x : x * 1,name = self.name)(input)
        return mux

    def get_config(self):
        config = {
                  'node_has_failed': self.has_failed,
                  'input1': self.input1,
                  'input2': self.input2,
                  'name': self.name
                }
        base_config = super(InputMux, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape