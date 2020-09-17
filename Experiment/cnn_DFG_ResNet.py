from __future__ import absolute_import, division, print_function

import os
import random
import warnings

import keras.backend as K
import keras.layers as layers
from Experiment.cnn_Vanilla_ResNet import (
    PARTITION_SETING,
    define_cnn_architecture_cloud,
    define_cnn_architecture_edge,
    define_cnn_architecture_fog,
    define_cnn_architecture_IoT,
    init_model,
)
from Experiment.common import compile_keras_parallel_model
from Experiment.common_CNN import (
    define_hyperconnection_weight_lambda_layers,
    set_hyperconnection_weights,
)
from keras_applications.imagenet_utils import (
    _obtain_input_shape,
    get_submodules_from_kwargs,
)


default_skip_hyperconnection_config = [1, 1]


def define_DFG_CNN_ResNet(
    input_shape=None,
    classes=10,
    block="basic",
    residual_unit="v2",
    repetitions=[2, 2, 2, 2],
    initial_filters=64,
    activation="softmax",
    include_top=True,
    input_tensor=None,
    dropout=None,
    transition_dilation_rate=(1, 1),
    initial_strides=(2, 2),
    initial_kernel_size=(7, 7),
    initial_pooling="max",
    final_pooling=None,
    top="classification",
    skip_hyperconnection_config=default_skip_hyperconnection_config,  # binary representating if a skip hyperconnection is alive [e1,IoT]
    reliability_setting=[1.0, 1.0],  # reliability of a node between 0 and 1 [f1,e1]
    hyperconnection_weights_scheme=1,
    num_gpus=1,
):

    (
        hyperconnection_weight_IoTe,
        hyperconnection_weight_IoTf,
        hyperconnection_weight_ef,
        hyperconnection_weight_ec,
        hyperconnection_weight_fc,
    ) = set_hyperconnection_weights(
        hyperconnection_weights_scheme, reliability_setting, skip_hyperconnection_config
    )
    (
        multiply_hyperconnection_weight_layer_IoTe,
        multiply_hyperconnection_weight_layer_IoTf,
        multiply_hyperconnection_weight_layer_ef,
        multiply_hyperconnection_weight_layer_ec,
        multiply_hyperconnection_weight_layer_fc,
    ) = define_hyperconnection_weight_lambda_layers(
        hyperconnection_weight_IoTe,
        hyperconnection_weight_IoTf,
        hyperconnection_weight_ef,
        hyperconnection_weight_ec,
        hyperconnection_weight_fc,
    )

    input_shape, block_fn, residual_unit = init_model(
        input_shape, classes, include_top, block, residual_unit, activation
    )
    img_input = layers.Input(shape=input_shape, tensor=input_tensor)

    # IoT Node
    iot_output, skip_iotfog = define_cnn_DFG_architecture_IoT(
        img_input, initial_filters, initial_kernel_size, initial_strides
    )
    # edge
    edge_output, skip_edgecloud, filters = define_cnn_DFG_architecture_edge(
        iot_output,
        repetitions[0],
        transition_dilation_rate,
        block_fn,
        initial_filters,
        dropout,
        residual_unit,
        initial_pooling,
        initial_strides,
        multiply_hyperconnection_weight_layer_IoTe,
    )

    # fog node
    fog_output, filters = define_cnn_DFG_architecture_fog(
        skip_iotfog,
        edge_output,
        repetitions[1],
        transition_dilation_rate,
        block_fn,
        filters,
        dropout,
        residual_unit,
        multiply_hyperconnection_weight_layer_IoTf,
        multiply_hyperconnection_weight_layer_ef,
    )

    # cloud node
    cloud_output = define_cnn_DFG_architecture_cloud(
        fog_output,
        skip_edgecloud,
        repetitions[2],
        repetitions[3],
        transition_dilation_rate,
        block_fn,
        filters,
        dropout,
        residual_unit,
        input_shape,
        classes,
        activation,
        include_top,
        top,
        final_pooling,
        multiply_hyperconnection_weight_layer_fc,
        multiply_hyperconnection_weight_layer_ec,
    )

    model, parallel_model = compile_keras_parallel_model(
        img_input, cloud_output, num_gpus
    )
    return model, parallel_model


def define_cnn_DFG_architecture_IoT(
    img_input, initial_filters, initial_kernel_size, initial_strides
):
    iot_output = define_cnn_architecture_IoT(
        img_input, initial_filters, initial_kernel_size, initial_strides
    )
    if PARTITION_SETING == 1:
        # need to go from (16,16,64) to (4,4,64) ????
        # 1x1 conv2d is used to change the filter size (from 64 to 64). Stride is 4 for 16->4
        # cifar-10
        skip_iotfog = layers.Conv2D(
            64, (1, 1), strides=4, use_bias=False, name="skip_hyperconnection_iotfog"
        )(iot_output)
    else:  # PARTITION_SETING == 2
        # need to go from (16,16,64) to (2,2,128) ????
        # 1x1 conv2d is used to change the filter size (from 64 to 128). Stride is 8 for 16->2
        # cifar-10
        skip_iotfog = layers.Conv2D(
            128, (1, 1), strides=8, use_bias=False, name="skip_hyperconnection_iotfog"
        )(iot_output)
    return iot_output, skip_iotfog


def define_cnn_DFG_architecture_edge(
    iot_output,
    r,
    transition_dilation_rate,
    block_fn,
    filters,
    dropout,
    residual_unit,
    initial_pooling,
    initial_strides,
    multiply_hyperconnection_weight_layer_IoTe=None,
    edge_failout_lambda=None,
):
    if multiply_hyperconnection_weight_layer_IoTe != None:
        iot_output = multiply_hyperconnection_weight_layer_IoTe(iot_output)
    edge_output, filters = define_cnn_architecture_edge(
        iot_output,
        r,
        transition_dilation_rate,
        block_fn,
        filters,
        dropout,
        residual_unit,
        initial_pooling,
        initial_strides,
    )
    if edge_failout_lambda != None:
        edge_output = edge_failout_lambda(edge_output)
    if PARTITION_SETING == 1:
        # need to go from (4,4,64) to (2,2,128) ????
        # 1x1 conv2d is used to change the filter size (from 64 to 128).  Stride is 2 for 4->2
        skip_edgecloud = layers.Conv2D(
            128,
            (1, 1),
            strides=2,
            use_bias=False,
            name="skip_hyperconnection_edgecloud",
        )(edge_output)
    else:  # PARTITION_SETING == 2
        # need to go from (2,2,128) to (1,1,256) ????
        # 1x1 conv2d is used to change the filter size (from 128 to 256).  Stride is 2 for 2->1
        skip_edgecloud = layers.Conv2D(
            256,
            (1, 1),
            strides=2,
            use_bias=False,
            name="skip_hyperconnection_edgecloud",
        )(edge_output)
    return edge_output, skip_edgecloud, filters


def define_cnn_DFG_architecture_fog(
    skip_iotfog,
    edge_output,
    r,
    transition_dilation_rate,
    block_fn,
    filters,
    dropout,
    residual_unit,
    multiply_hyperconnection_weight_layer_IoTf=None,
    multiply_hyperconnection_weight_layer_ef=None,
):
    if (
        multiply_hyperconnection_weight_layer_IoTf == None
        or multiply_hyperconnection_weight_layer_ef == None
    ):
        fog_input = layers.add([skip_iotfog, edge_output], name="node2_input")
    else:
        fog_input = layers.add(
            [
                multiply_hyperconnection_weight_layer_IoTf(skip_iotfog),
                multiply_hyperconnection_weight_layer_ef(edge_output),
            ],
            name="node2_input",
        )
    fog_output, filters = define_cnn_architecture_fog(
        fog_input,
        r,
        transition_dilation_rate,
        block_fn,
        filters,
        dropout,
        residual_unit,
    )
    return fog_output, filters


def define_cnn_DFG_architecture_cloud(
    fog_output,
    skip_edgecloud,
    r1,
    r2,
    transition_dilation_rate,
    block_fn,
    filters,
    dropout,
    residual_unit,
    input_shape,
    classes,
    activation,
    include_top,
    top,
    final_pooling,
    multiply_hyperconnection_weight_layer_fc=None,
    multiply_hyperconnection_weight_layer_ec=None,
):
    if (
        multiply_hyperconnection_weight_layer_fc == None
        or multiply_hyperconnection_weight_layer_ec == None
    ):
        cloud_input = layers.add([fog_output, skip_edgecloud], name="node1_input")
    else:
        cloud_input = layers.add(
            [
                multiply_hyperconnection_weight_layer_fc(fog_output),
                multiply_hyperconnection_weight_layer_ec(skip_edgecloud),
            ],
            name="node1_input",
        )
    cloud_output = define_cnn_architecture_cloud(
        cloud_input,
        r1,
        r2,
        transition_dilation_rate,
        block_fn,
        filters,
        dropout,
        residual_unit,
        input_shape,
        classes,
        activation,
        include_top,
        top,
        final_pooling,
    )
    return cloud_output
