import random

import keras.layers as layers
from Experiment.cnn_DFG_MobileNet import define_DFG_CNN_MobileNet
from Experiment.cnn_DFG_ResNet import define_DFG_CNN_ResNet
from Experiment.cnn_ResiliNet_MobileNet import (
    MUX_ADDS as MUX_ADDS_MobileNet,
    define_ResiliNet_CNN_MobileNet,
)
from Experiment.cnn_ResiliNet_ResNet import (
    MUX_ADDS as MUX_ADDS_ResNet,
    define_ResiliNet_CNN_ResNet,
)
from Experiment.cnn_Vanilla_MobileNet import define_vanilla_CNN_MobileNet
from Experiment.cnn_Vanilla_ResNet import define_vanilla_CNN_ResNet
from Experiment.custom_ops import Failout


def define_model(
    iteration,
    model_name,
    dataset_name,
    input_shape,
    classes,
    alpha,
    strides,
    num_gpus,
    weights,
):
    # ResiliNet
    if model_name == "ResiliNet":
        if dataset_name == "cifar_resnet":
            mux_adds_str = "mux_adds" if MUX_ADDS_ResNet else ""
            model, parallel_model = define_ResiliNet_CNN_ResNet(
                input_shape=input_shape,
                classes=classes,
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
                num_gpus=num_gpus,
            )
        else:
            mux_adds_str = "mux_adds" if MUX_ADDS_MobileNet else ""
            model, parallel_model = define_ResiliNet_CNN_MobileNet(
                classes=classes,
                input_shape=input_shape,
                alpha=alpha,
                strides=strides,
                num_gpus=num_gpus,
                weights=weights,
            )
        model_file = (
            "models/"
            + dataset_name
            + str(iteration)
            + mux_adds_str
            + "average_accuracy_ResiliNet.h5"
        )
    # DFG
    if model_name == "DFG":
        if dataset_name == "cifar_resnet":
            model, parallel_model = define_DFG_CNN_ResNet(
                input_shape=input_shape,
                classes=classes,
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
                num_gpus=num_gpus,
            )
        else:
            model, parallel_model = define_DFG_CNN_MobileNet(
                classes=classes,
                input_shape=input_shape,
                alpha=alpha,
                strides=strides,
                num_gpus=num_gpus,
                weights=weights,
            )
        model_file = (
            "models/" + dataset_name + str(iteration) + "average_accuracy_DFG.h5"
        )
    # Vanilla model
    if model_name == "Vanilla":
        if dataset_name == "cifar_resnet":
            model, parallel_model = define_vanilla_CNN_ResNet(
                input_shape=input_shape,
                classes=classes,
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
                num_gpus=num_gpus,
            )
        else:
            model, parallel_model = define_vanilla_CNN_MobileNet(
                classes=classes,
                input_shape=input_shape,
                alpha=alpha,
                strides=strides,
                num_gpus=num_gpus,
                weights=weights,
            )
        model_file = (
            "models/" + dataset_name + str(iteration) + "average_accuracy_vanilla.h5"
        )

    return model, parallel_model, model_file


def set_hyperconnection_weights(
    hyperconnection_weights_scheme, reliability_setting, skip_hyperconnection_config
):
    # weighted by 1
    if hyperconnection_weights_scheme == 1:
        hyperconnection_weight_IoTe = 1
        hyperconnection_weight_IoTf = 1
        hyperconnection_weight_ef = 1
        hyperconnection_weight_ec = 1
        hyperconnection_weight_fc = 1
    # normalized reliability
    elif hyperconnection_weights_scheme == 2:
        hyperconnection_weight_IoTe = 1
        hyperconnection_weight_IoTf = 1 / (1 + reliability_setting[1])
        hyperconnection_weight_ef = reliability_setting[1] / (
            1 + reliability_setting[1]
        )
        hyperconnection_weight_ec = reliability_setting[1] / (
            reliability_setting[0] + reliability_setting[1]
        )
        hyperconnection_weight_fc = reliability_setting[0] / (
            reliability_setting[0] + reliability_setting[1]
        )
    # reliability
    elif hyperconnection_weights_scheme == 3:
        hyperconnection_weight_IoTe = 1
        hyperconnection_weight_IoTf = 1
        hyperconnection_weight_ef = reliability_setting[1]
        hyperconnection_weight_ec = reliability_setting[1]
        hyperconnection_weight_fc = reliability_setting[0]
    # randomly weighted between 0 and 1
    elif hyperconnection_weights_scheme == 4:
        hyperconnection_weight_IoTe = random.uniform(0, 1)
        hyperconnection_weight_IoTf = random.uniform(0, 1)
        hyperconnection_weight_ef = random.uniform(0, 1)
        hyperconnection_weight_ec = random.uniform(0, 1)
        hyperconnection_weight_fc = random.uniform(0, 1)
    # randomly weighted between 0 and 10
    elif hyperconnection_weights_scheme == 5:
        hyperconnection_weight_IoTe = random.uniform(0, 10)
        hyperconnection_weight_IoTf = random.uniform(0, 10)
        hyperconnection_weight_ef = random.uniform(0, 10)
        hyperconnection_weight_ec = random.uniform(0, 10)
        hyperconnection_weight_fc = random.uniform(0, 10)
    else:
        raise ValueError("Incorrect scheme value")
    (
        hyperconnection_weight_IoTf,
        hyperconnection_weight_ec,
    ) = remove_skip_hyperconnection_for_sensitvity_experiment(
        skip_hyperconnection_config,
        hyperconnection_weight_IoTf,
        hyperconnection_weight_ec,
    )
    return (
        hyperconnection_weight_IoTe,
        hyperconnection_weight_IoTf,
        hyperconnection_weight_ef,
        hyperconnection_weight_ec,
        hyperconnection_weight_fc,
    )


def remove_skip_hyperconnection_for_sensitvity_experiment(
    skip_hyperconnection_config, connection_weight_IoTf, connection_weight_ec
):
    # take away the skip hyperconnection if the value in hyperconnections array is 0
    # from edge to cloud
    if skip_hyperconnection_config[0] == 0:
        connection_weight_ec = 0
    # from iot to fog
    if skip_hyperconnection_config[1] == 0:
        connection_weight_IoTf = 0
    return connection_weight_IoTf, connection_weight_ec


def define_hyperconnection_weight_lambda_layers(
    hyperconnection_weight_IoTe,
    hyperconnection_weight_IoTf,
    hyperconnection_weight_ef,
    hyperconnection_weight_ec,
    hyperconnection_weight_fc,
):
    # define lambdas for multiplying node weights by connection weight
    multiply_hyperconnection_weight_layer_IoTe = layers.Lambda(
        (lambda x: x * hyperconnection_weight_IoTe), name="connection_weight_IoTe"
    )
    multiply_hyperconnection_weight_layer_IoTf = layers.Lambda(
        (lambda x: x * hyperconnection_weight_IoTf), name="connection_weight_IoTf"
    )
    multiply_hyperconnection_weight_layer_ef = layers.Lambda(
        (lambda x: x * hyperconnection_weight_ef), name="connection_weight_ef"
    )
    multiply_hyperconnection_weight_layer_ec = layers.Lambda(
        (lambda x: x * hyperconnection_weight_ec), name="connection_weight_ec"
    )
    multiply_hyperconnection_weight_layer_fc = layers.Lambda(
        (lambda x: x * hyperconnection_weight_fc), name="connection_weight_fc"
    )
    return (
        multiply_hyperconnection_weight_layer_IoTe,
        multiply_hyperconnection_weight_layer_IoTf,
        multiply_hyperconnection_weight_layer_ef,
        multiply_hyperconnection_weight_layer_ec,
        multiply_hyperconnection_weight_layer_fc,
    )


def cnn_failout_definitions(failout_survival_setting):
    fog_reliability = failout_survival_setting[0]
    edge_reliability = failout_survival_setting[1]

    edge_failout_lambda = Failout(edge_reliability)
    fog_failout_lambda = Failout(fog_reliability)
    return edge_failout_lambda, fog_failout_lambda
