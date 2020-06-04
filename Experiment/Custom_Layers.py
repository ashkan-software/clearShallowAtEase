from keras.layers import Layer, Add, Lambda
import keras.layers as layers
import keras.backend as K
import tensorflow as tf

class Failout(Layer):
    """Applies Failout to the output of a node.
    # Arguments
        failout_survival_rate: float between 0 and 1. Probability of survival of a node (1 - prob_failure).
        seed: A Python integer to use as random seed.
    """
    def __init__(self, failout_survival_rate, seed=None, **kwargs):
        super(Failout, self).__init__(**kwargs)
        self.seed = seed
        self.failout_survival_rate = K.variable(failout_survival_rate)
        self.has_failed = None

    def call(self, inputs):
        rand = K.random_uniform(K.variable(0).shape, seed = self.seed)
        # assumes that there is only one input in inputs
        fail = Lambda(lambda x: x * 0)
        self.has_failed = K.greater(rand, self.failout_survival_rate)
        failout_ready_inputs = K.switch(self.has_failed, fail(inputs), inputs)
        failout = K.in_train_phase(failout_ready_inputs, inputs)
        # failout = tf.print(failout, [failout], message='output of failout layer=', summarize=100)
        return failout

    def get_config(self):
        config = {
                  'seed': self.seed,
                  'failout_survival_rate': self.failout_survival_rate,
                  'has_failed': self.has_failed
                }
        base_config = super(Failout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class InputMux(Add):
    """
    Input Multiplexer for a node that receives input from more than one downstream nodes, 
    # Arguments
        node_has_failed: Boolean Tensor showing if the downstream node has failed
    """
    def __init__(self, has_failed, **kwargs):
        super(InputMux, self).__init__(**kwargs)
        self.has_failed = has_failed
        # self.name = name

    def _merge_function(self, inputs):
        """
        # inputs
        the two incoming connections to a node. inputs[0] MUST be the input from skip hyperconnection
        and inputs[1] MUST be the input from the node below.
        
        Since Failout happens during training, we do not need the Failout informaton for selection of
        the inputs in `K.switch(self.has_failed, inputs[0], inputs[1])`. During training, the two
        inpus are always added (Failout during training will make the output zero, so it has its
        own impact on the training.)
        """
        
        selected = K.switch(self.has_failed, inputs[0], inputs[1]) # selects one of the inputs. 
        # If the node below has failed, use the input from skip hyperconnection, otherwise, use the input from the node below
        
        added = layers.add(inputs) # calls the add function

        output = K.in_train_phase(added, selected)
        return output