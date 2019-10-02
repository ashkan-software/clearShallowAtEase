
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from Experiment.MobileNet_blocks import _conv_block, _depthwise_conv_block
import os
import warnings
#from tensorflow import keras
import keras
import keras.backend as K
import keras.layers as layers
from keras.backend import zeros
from keras_applications.imagenet_utils import _obtain_input_shape, get_submodules_from_kwargs
from keras_applications import imagenet_utils
 
from Experiment.cnn_deepFogGuard import define_cnn_deepFogGuard_architecture_IoT, define_cnn_deepFogGuard_architecture_cloud, define_cnn_deepFogGuard_architecture_edge, define_cnn_deepFogGuard_architecture_fog
from Experiment.Failout import Failout
# ResiliNet
def define_ResiliNet_CNN(input_shape=None,
                                alpha=1.0,
                                depth_multiplier=1,
                                include_top=True,
                                pooling=None,
                                classes=1000, 
                                strides = (2,2),
                                failout_survival_setting = [1.0,1.0],
                                **kwargs):
    """Instantiates the MobileNet architecture.

    # Arguments
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)`
            (with `channels_last` data format)
            or (3, 224, 224) (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        alpha: controls the width of the network. This is known as the
            width multiplier in the MobileNet paper.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        depth_multiplier: depth multiplier for depthwise convolution. This
            is called the resolution multiplier in the MobileNet paper.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
    """
    
    # Determine proper input shape and default size.
    img_input = layers.Input(shape=input_shape)  

    # failout definitions
    edge_failure_lambda, fog_failure_lambda = cnn_failout_definitions(failout_survival_setting)

     # iot node
    iot_output,skip_iotfog = define_cnn_deepFogGuard_architecture_IoT(input_shape,alpha,img_input, strides = strides)
    
    # edge node
    edge_output, skip_edgecloud = define_cnn_deepFogGuard_architecture_edge(iot_output,alpha, depth_multiplier, strides = strides, edge_failure_lambda = edge_failure_lambda)
    # edge_output = edge_failure_lambda(edge_output)
    
    # fog node
    fog_output = define_cnn_deepFogGuard_architecture_fog(skip_iotfog, edge_output, alpha, depth_multiplier, strides = strides)
    fog_output = fog_failure_lambda(fog_output)

    # cloud node
    cloud_output = define_cnn_deepFogGuard_architecture_cloud(fog_output, skip_edgecloud, alpha, depth_multiplier, classes, include_top, pooling)
    
    # Create model.
    model = keras.Model(img_input, cloud_output, name='ANRL_mobilenet')
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
def define_deepFogGuardPlus_CNN(input_shape=None,
              alpha=1.0,
              depth_multiplier=1,
              dropout=0,
              include_top=True,
              weights='imagenet',
              input_tensor=None,
              pooling=None,
              classes=1000, 
              survive_rates = [1,1],
              **kwargs):
    """Instantiates the MobileNet architecture.
    # Arguments
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)`
            (with `channels_last` data format)
            or (3, 224, 224) (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        alpha: controls the width of the network. This is known as the
            width multiplier in the MobileNet paper.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        depth_multiplier: depth multiplier for depthwise convolution. This
            is called the resolution multiplier in the MobileNet paper.
        dropout: dropout rate
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        survive_rates: survival rates of network nodes, default value is [1,1]
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
    """

    img_input = layers.Input(shape=input_shape)
    # # variables for node dropout

    # edge_survive_rate = K.variable(survive_rates[0])
    # fog_survive_rate = K.variable(survive_rates[1])

    # edge_survive_rate = K.print_tensor(edge_survive_rate, "edge_survive_rate is ")
    # fog_survive_rate = K.print_tensor(fog_survive_rate, "fog_survive_rate is ")
    # # set training phase to true 
    # # seeds so the random_number is different for each fog node 
    # edge_rand = K.in_train_phase(K.random_uniform(shape=rand.shape),rand)
    # fog_rand = K.in_train_phase(K.random_uniform(shape=rand.shape),rand)

    # edge_rand = K.print_tensor(edge_rand, "edge_rand is ")
    # fog_rand = K.print_tensor(fog_rand, "fog_rand is ")
    #  # define lambda for failure, only fail during training
    # edge_failure_lambda = layers.Lambda(lambda x : K.switch(K.greater(edge_rand,edge_survive_rate), x * 0, x),name = 'edge_failure_lambda')
    # fog_failure_lambda = layers.Lambda(lambda x : K.switch(K.greater(fog_rand,fog_survive_rate), x * 0, x),name = 'fog_failure_lambda')

    # variables for node dropout
#    edge_rand = K.variable(0)
#    fog_rand = K.variable(0)
#    edge_survive_rate = K.variable(survive_rates[0])
#    fog_survive_rate = K.variable(survive_rates[1])
#    edge_survive_rate = K.print_tensor(edge_survive_rate, "edge_survive_rate is ")
#    fog_survive_rate = K.print_tensor(fog_survive_rate, "fog_survive_rate is ")
    # set training phase to true 
#    K.set_learning_phase(1)
    print('ssssssssssssssssssssss',K.learning_phase())
#    if K.learning_phase():
        # seeds so the random_number is different for each fog node 
#        edge_rand = K.random_uniform(shape=edge_rand.shape,seed=7)
#        fog_rand = K.random_uniform(shape=fog_rand.shape,seed=11)
#    edge_rand = K.print_tensor(edge_rand, "edge_rand is ")
#        fog_rand = K.print_tensor(fog_rand, "fog_rand is ")
    # define lambda for failure, only fail during training
#    edge_failure_lambda = layers.Lambda(lambda x : K.switch(K.greater(edge_rand,edge_survive_rate), x * 0, x),name = 'edge_failure_lambda')
#    fog_failure_lambda = layers.Lambda(lambda x : K.switch(K.greater(fog_rand,fog_survive_rate), x * 0, x),name = 'fog_failure_lambda')
 
    edge_failure_lambda, fog_failure_lambda = cnn_failout_definitions(survive_rates)
  
    # changed the strides from 2 to 1 since cifar-10 images are smaller
    # IoT node
    iot = _conv_block(img_input, 32, alpha, strides=(1, 1)) # size: (31,31,16)
    connection_iotfog = layers.Conv2D(64,(1,1),strides = 1, use_bias = False, name = "skip_hyperconnection_iotfog")(iot)
 
    # edge 
    edge = _depthwise_conv_block(iot, 64, alpha, depth_multiplier, block_id=1)

    edge = _depthwise_conv_block(edge, 128, alpha, depth_multiplier,
                              strides=(1, 1), block_id=2)
    connection_edgefog = _depthwise_conv_block(edge, 128, alpha, depth_multiplier, block_id=3) # size:  (None, 31, 31, 64) 
    connection_edgefog = edge_failure_lambda(connection_edgefog)
    # skip hyperconnection, used 1x1 convolution to project shape of node output into (7,7,256)\
    # check it back to normal skip_hyperconnection
    connection_edgecloud = layers.Conv2D(256,(1,1),strides = 4, use_bias = False, name = "skip_hyperconnection_edgecloud")(connection_edgefog)
    connection_fog = layers.add([connection_iotfog,connection_edgefog], name = "connection_fog")
    
    # fog node
    fog = _depthwise_conv_block(connection_fog, 256, alpha, depth_multiplier, # size: (None, 32, 32, 64)
                              strides=(2, 2), block_id=4)
    fog = _depthwise_conv_block(fog, 256, alpha, depth_multiplier, block_id=5)

    fog = _depthwise_conv_block(fog, 512, alpha, depth_multiplier,
                              strides=(2, 2), block_id=6)
    fog = _depthwise_conv_block(fog, 512, alpha, depth_multiplier, block_id=7)
    fog = _depthwise_conv_block(fog, 512, alpha, depth_multiplier, block_id=8) #size : (None, 7, 7, 256) 
    # pad from (7,7,256) to (8,8,256)
    connection_fogcloud = layers.ZeroPadding2D(padding = ((0, 1), (0, 1)), name = "fogcloud_connection_padding")(fog)
    connection_fogcloud = fog_failure_lambda(connection_fogcloud)
    connection_cloud = layers.add([connection_fogcloud,connection_edgecloud], name = "Cloud_Input")

    # cloud node
    cloud = _depthwise_conv_block(connection_cloud, 512, alpha, depth_multiplier, block_id=9) # size: (None, 7, 7, 256)
    cloud = _depthwise_conv_block(cloud, 512, alpha, depth_multiplier, block_id=10)
    cloud = _depthwise_conv_block(cloud, 512, alpha, depth_multiplier, block_id=11)

    cloud = _depthwise_conv_block(cloud, 1024, alpha, depth_multiplier,
                              strides=(2, 2), block_id=12)
    cloud = _depthwise_conv_block(cloud, 1024, alpha, depth_multiplier, block_id=13)

    if include_top:
        if K.image_data_format() == 'channels_first':
            shape = (int(1024 * alpha), 1, 1)
        else:
            shape = (1, 1, int(1024 * alpha))

        cloud = layers.GlobalAveragePooling2D()(cloud)
        cloud = layers.Reshape(shape, name='reshape_1')(cloud)
        # dropout is only in the cloud node, we can potentially use it
        cloud = layers.Dropout(dropout, name='dropout')(cloud)
        cloud = layers.Conv2D(classes, (1, 1),
                          padding='same',
                          name='conv_preds')(cloud)
        cloud = layers.Reshape((classes,), name='reshape_2')(cloud)
        cloud = layers.Activation('softmax', name='output')(cloud)
    else:
        if pooling == 'avg':
            cloud = layers.GlobalAveragePooling2D()(cloud)
        elif pooling == 'max':
            cloud = layers.GlobalMaxPooling2D()(cloud)
    # Create model.
    model = keras.Model(img_input, cloud, name="ANRL_mobilenet")
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def cnn_failout_definitions(failout_survival_setting):
    edge_reliability = failout_survival_setting[0]
    fog_reliability = failout_survival_setting[1]
    

    edge_failure_lambda = Failout(edge_reliability)
    fog_failure_lambda = Failout(fog_reliability)
    return edge_failure_lambda, fog_failure_lambda

    # variables for node dropout
#    edge_rand = K.variable(0)
#    fog_rand = K.variable(0)
#    edge_survive_rate = K.variable(failout_survival_setting[0])
#    fog_survive_rate = K.variable(failout_survival_setting[1])
    # set training phase to true 
#    K.set_learning_phase(1)
#    if K.learning_phase():
        # seeds so the random_number is different for each fog node 
#        edge_rand = K.random_uniform(shape=edge_rand.shape,seed=7)
#        fog_rand = K.random_uniform(shape=fog_rand.shape,seed=11)
     # define lambda for failure, only fail during training
#    edge_failure_lambda = layers.Lambda(lambda x : K.switch(K.greater(edge_rand,edge_survive_rate), x * 0, x),name = 'edge_failure_lambda')
#    fog_failure_lambda = layers.Lambda(lambda x : K.switch(K.greater(fog_rand,fog_survive_rate), x * 0, x),name = 'fog_failure_lambda')
#    return edge_failure_lambda, fog_failure_lambda
