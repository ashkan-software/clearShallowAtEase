from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import six
import warnings

import keras.backend as K
import keras.layers as layers
from keras_applications.imagenet_utils import _obtain_input_shape
from Experiment.ResNet_blocks import _handle_dim_ordering, basic_block, _string_to_function, bottleneck, _bn_relu_conv, _conv_bn_relu, _residual_block, _bn_relu
from Experiment.common_exp_methods import compile_keras_parallel_model

BASE_WEIGHT_PATH = ('https://github.com/fchollet/deep-learning-models/'
                    'releases/download/v0.6/')
def define_vanilla_CNN_ResNet(input_shape=None, classes=10, block='bottleneck', residual_unit='v2',
                            repetitions=[2, 2, 2, 2], initial_filters=64, activation='softmax', include_top=True,
                            input_tensor=None, dropout=None, transition_dilation_rate=(1, 1),
                            initial_strides=(2, 2), initial_kernel_size=(7, 7), initial_pooling='max',
                            final_pooling=None, top='classification', num_gpus = 1):
    """Builds a custom ResNet18 architecture.
    Args:
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` dim ordering)
            or `(3, 224, 224)` (with `channels_first` dim ordering).
            It should have exactly 3 dimensions,
            and width and height should be no smaller than 8.
            E.g. `(224, 224, 3)` would be one valid value.
        classes: The number of outputs at final softmax layer
        block: The block function to use. This is either `'basic'` or `'bottleneck'`.
            The original paper used `basic` for layers < 50.
        repetitions: Number of repetitions of various block units.
            At each block unit, the number of filters are doubled and the input size
            is halved. 
        residual_unit: the basic residual unit, 'v1' for conv bn relu, 'v2' for bn relu
            conv. See [Identity Mappings in
            Deep Residual Networks](https://arxiv.org/abs/1603.05027)
            for details.
        dropout: None for no dropout, otherwise rate of dropout from 0 to 1.
            Based on [Wide Residual Networks.(https://arxiv.org/pdf/1605.07146) paper.
        transition_dilation_rate: Dilation rate for transition layers. For semantic
            segmentation of images use a dilation rate of (2, 2).
        initial_strides: Stride of the very first residual unit and MaxPooling2D call,
            with default (2, 2), set to (1, 1) for small images like cifar.
        initial_kernel_size: kernel size of the very first convolution, (7, 7) for
            imagenet and (3, 3) for small image datasets like tiny imagenet and cifar.
            See [ResNeXt](https://arxiv.org/abs/1611.05431) paper for details.
        initial_pooling: Determine if there will be an initial pooling layer,
            'max' for imagenet and None for small image datasets.
            See [ResNeXt](https://arxiv.org/abs/1611.05431) paper for details.
        final_pooling: Optional pooling mode for feature extraction at the final
            model layer when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        top: Defines final layers to evaluate based on a specific problem type. Options
            are 'classification' for ImageNet style problems, 'segmentation' for
            problems like the Pascal VOC dataset, and None to exclude these layers
            entirely.
    Returns:
        The keras `Model`.
    """

    input_shape, block_fn, residual_unit = init_model(input_shape, classes, include_top, block, residual_unit, activation)
    img_input = layers.Input(shape=input_shape, tensor=input_tensor) 
    
    # IoT Node
    iot = define_cnn_architecture_IoT(img_input, initial_filters, initial_kernel_size, initial_strides)
    # edge 
    edge, filters = define_cnn_architecture_edge(iot, repetitions[0], transition_dilation_rate, block_fn, initial_filters, dropout, residual_unit, initial_pooling, initial_strides)
    
    # fog node
    fog = layers.Lambda(lambda x : x * 1,name = 'node2_input')(edge)
    fog, filters = define_cnn_architecture_fog(fog, repetitions[1], transition_dilation_rate, block_fn, filters, dropout, residual_unit)
    
    # cloud node
    cloud = layers.Lambda(lambda x : x * 1,name = 'node1_input')(fog)
    cloud = define_cnn_architecture_cloud(cloud, repetitions[2], repetitions[3], transition_dilation_rate, block_fn, filters, dropout, residual_unit, input_shape, classes, activation, include_top, top, final_pooling)

    model, parallel_model = compile_keras_parallel_model(img_input, cloud, num_gpus)
    return model, parallel_model

def init_model(input_shape, classes, include_top, block, residual_unit, activation):
    if activation not in ['softmax', 'sigmoid', None]:
        raise ValueError('activation must be one of "softmax", "sigmoid", or None')
    if activation == 'sigmoid' and classes != 1:
        raise ValueError('sigmoid activation can only be used when classes = 1')
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=32,
                                      min_size=8,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top)
    _handle_dim_ordering()
    if len(input_shape) != 3:
        raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

    if block == 'basic':
        block_fn = basic_block
    elif block == 'bottleneck':
        block_fn = bottleneck
    elif isinstance(block, six.string_types):
        block_fn = _string_to_function(block)
    else:
        block_fn = block

    if residual_unit == 'v2':
        residual_unit = _bn_relu_conv
    elif residual_unit == 'v1':
        residual_unit = _conv_bn_relu
    elif isinstance(residual_unit, six.string_types):
        residual_unit = _string_to_function(residual_unit)
    else:
        residual_unit = residual_unit

    # Permute dimension order if necessary
    if K.image_data_format() == 'channels_first':
        input_shape = (input_shape[1], input_shape[2], input_shape[0])
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=32,
                                      min_size=8,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top)

    return input_shape, block_fn, residual_unit                              

def define_cnn_architecture_IoT(img_input,initial_filters, initial_kernel_size, initial_strides):
    return _conv_bn_relu(filters=initial_filters, kernel_size=initial_kernel_size,
                      strides=initial_strides)(img_input)

def define_cnn_architecture_edge(iot_output, r, transition_dilation_rate, block_fn, filters, dropout, residual_unit, initial_pooling, initial_strides):
    if initial_pooling == 'max':
        edge = layers.MaxPooling2D(pool_size=(3, 3), strides=initial_strides, padding="same")(iot_output)
    edge_output, filters = _helper_define_conv_blocks(edge, 0, r, transition_dilation_rate, block_fn, filters, dropout, residual_unit)
    return edge_output, filters

def define_cnn_architecture_fog(edge_output, r, transition_dilation_rate, block_fn, filters, dropout, residual_unit):
    fog_output, filters = _helper_define_conv_blocks(edge_output, 1, r, transition_dilation_rate, block_fn, filters, dropout, residual_unit)
    return fog_output, filters

def define_cnn_architecture_cloud(fog_output, r1, r2, transition_dilation_rate, block_fn, filters, dropout, residual_unit, input_shape, classes, activation, include_top, top, final_pooling):
    cloud, filters = _helper_define_conv_blocks(fog_output, 2, r1, transition_dilation_rate, block_fn, filters, dropout, residual_unit)
    cloud, filters = _helper_define_conv_blocks(cloud, 3, r2, transition_dilation_rate, block_fn, filters, dropout, residual_unit)
      # Last activation
    cloud = _bn_relu(cloud)
    # Classifier block
    if include_top and top is 'classification':
        cloud = layers.GlobalAveragePooling2D()(cloud)
        cloud = layers.Dense(units=classes, activation=activation,
                  kernel_initializer="he_normal")(cloud)
    elif include_top and top is 'segmentation':
        cloud = layers.Conv2D(classes, (1, 1), activation='linear', padding='same')(cloud)

        if K.image_data_format() == 'channels_first':
            channel, row, col = input_shape
        else:
            row, col, channel = input_shape

        cloud = layers.Reshape((row * col, classes))(cloud)
        cloud = layers.Activation(activation)(cloud)
        cloud = layers.Reshape((row, col, classes))(cloud)
    elif final_pooling == 'avg':
        cloud = layers.GlobalAveragePooling2D()(cloud)
    elif final_pooling == 'max':
        cloud = layers.GlobalMaxPooling2D()(cloud)
    return cloud

def _helper_define_conv_blocks(input, i, r, transition_dilation_rate, block_fn, filters, dropout, residual_unit):
    transition_dilation_rates = [transition_dilation_rate] * r
    transition_strides = [(1, 1)] * r
    if transition_dilation_rate == (1, 1):
        transition_strides[0] = (2, 2)
    output = _residual_block(block_fn, filters=filters,
                            stage=i, blocks=r,
                            is_first_layer=(i == 0),
                            dropout=dropout,
                            transition_dilation_rates=transition_dilation_rates,
                            transition_strides=transition_strides,
                            residual_unit=residual_unit)(input)
    filters *= 2
    return output, filters