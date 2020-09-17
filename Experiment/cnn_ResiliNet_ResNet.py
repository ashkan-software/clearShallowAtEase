from __future__ import absolute_import, division, print_function

import os
import warnings

import keras.backend as K
import keras.layers as layers
from Experiment.cnn_DFG_ResNet import (
    default_skip_hyperconnection_config,
    define_cnn_DFG_architecture_edge,
    define_cnn_DFG_architecture_IoT,
    init_model,
)
from Experiment.cnn_Vanilla_ResNet import (
    define_cnn_architecture_cloud,
    define_cnn_architecture_fog,
)
from Experiment.common import compile_keras_parallel_model
from Experiment.common_CNN import (
    cnn_failout_definitions,
    define_hyperconnection_weight_lambda_layers,
    set_hyperconnection_weights,
)
from Experiment.custom_ops import InputMux
from keras.layers import Lambda


MUX_ADDS = False


def define_ResiliNet_CNN_ResNet(
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
    failout_survival_setting=[0.9, 0.9],
    skip_hyperconnection_config=default_skip_hyperconnection_config,
    reliability_setting=[1.0, 1.0],
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

    # failout definitions
    edge_failout_lambda, fog_failout_lambda = cnn_failout_definitions(
        failout_survival_setting
    )

    # IoT Node
    iot_output, skip_iotfog = define_cnn_ResiliNet_architecture_IoT(
        img_input, initial_filters, initial_kernel_size, initial_strides
    )
    # edge
    edge_output, skip_edgecloud, filters = define_cnn_ResiliNet_architecture_edge(
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
        edge_failout_lambda=edge_failout_lambda,
    )

    # fog node
    fog_output, filters = define_cnn_ResiliNet_architecture_fog(
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
    fog_output = fog_failout_lambda(fog_output)

    # cloud node
    cloud_output = define_cnn_ResiliNet_architecture_cloud(
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


def define_cnn_ResiliNet_architecture_IoT(
    img_input, initial_filters, initial_kernel_size, initial_strides
):
    iot_output, skip_iotfog = define_cnn_DFG_architecture_IoT(
        img_input, initial_filters, initial_kernel_size, initial_strides
    )
    return iot_output, skip_iotfog


def define_cnn_ResiliNet_architecture_edge(
    iot_output,
    r,
    transition_dilation_rate,
    block_fn,
    filters,
    dropout,
    residual_unit,
    initial_pooling,
    initial_strides,
    multiply_hyperconnection_weight_layer_IoTe,
    edge_failout_lambda,
):
    edge_output, skip_edgecloud, filters = define_cnn_DFG_architecture_edge(
        iot_output,
        r,
        transition_dilation_rate,
        block_fn,
        filters,
        dropout,
        residual_unit,
        initial_pooling,
        initial_strides,
        multiply_hyperconnection_weight_layer_IoTe,
        edge_failout_lambda,
    )
    return edge_output, skip_edgecloud, filters


def define_cnn_ResiliNet_architecture_fog(
    skip_iotfog,
    edge_output,
    r,
    transition_dilation_rate,
    block_fn,
    filters,
    dropout,
    residual_unit,
    multiply_hyperconnection_weight_layer_IoTf,
    multiply_hyperconnection_weight_layer_ef,
):
    if (
        multiply_hyperconnection_weight_layer_IoTf == None
        or multiply_hyperconnection_weight_layer_ef == None
    ):
        fog_input = Lambda(InputMux(MUX_ADDS), name="node2_input")(
            [skip_iotfog, edge_output]
        )
    else:
        fog_input = Lambda(InputMux(MUX_ADDS), name="node2_input")(
            [
                multiply_hyperconnection_weight_layer_IoTf(skip_iotfog),
                multiply_hyperconnection_weight_layer_ef(edge_output),
            ]
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


def define_cnn_ResiliNet_architecture_cloud(
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
    multiply_hyperconnection_weight_layer_fc,
    multiply_hyperconnection_weight_layer_ec,
):
    if (
        multiply_hyperconnection_weight_layer_fc == None
        or multiply_hyperconnection_weight_layer_ec == None
    ):
        cloud_input = Lambda(InputMux(MUX_ADDS), name="node1_input")(
            [skip_edgecloud, fog_output]
        )
    else:
        cloud_input = Lambda(InputMux(MUX_ADDS), name="node1_input")(
            [
                multiply_hyperconnection_weight_layer_ec(skip_edgecloud),
                multiply_hyperconnection_weight_layer_fc(fog_output),
            ]
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
