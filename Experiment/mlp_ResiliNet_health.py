import random

import keras.backend as K
import keras.layers as layers
from Experiment.custom_ops import Failout, InputMux
from Experiment.mlp_DFG_health import (
    default_skip_hyperconnection_config,
    define_hyperconnection_weight_lambda_layers,
    define_MLP_DFG_architecture_edge,
    define_MLP_DFG_architecture_IoT,
    set_hyperconnection_weights,
)
from Experiment.mlp_Vanilla_health import (
    define_MLP_architecture_cloud,
    define_MLP_architecture_fog1,
    define_MLP_architecture_fog2,
)
from keras.backend import constant
from keras.layers import Activation, Dense, Input, Lambda
from keras.models import Model, Sequential


ResiliNetPlus = False


def define_ResiliNet_MLP(
    num_vars,
    num_classes,
    hidden_units,
    failout_survival_setting=[0.9, 0.9, 0.9],
    reliability_setting=[1.0, 1.0, 1.0],
    skip_hyperconnection_config=default_skip_hyperconnection_config,
    hyperconnection_weights_scheme=1,
):

    (
        hyperconnection_weight_IoTe,
        hyperconnection_weight_IoTf2,
        hyperconnection_weight_ef2,
        hyperconnection_weight_ef1,
        hyperconnection_weight_f2f1,
        hyperconnection_weight_f2c,
        hyperconnection_weight_f1c,
    ) = set_hyperconnection_weights(
        hyperconnection_weights_scheme, reliability_setting, skip_hyperconnection_config
    )
    (
        multiply_hyperconnection_weight_layer_IoTe,
        multiply_hyperconnection_weight_layer_IoTf2,
        multiply_hyperconnection_weight_layer_ef2,
        multiply_hyperconnection_weight_layer_ef1,
        multiply_hyperconnection_weight_layer_f2f1,
        multiply_hyperconnection_weight_layer_f2c,
        multiply_hyperconnection_weight_layer_f1c,
    ) = define_hyperconnection_weight_lambda_layers(
        hyperconnection_weight_IoTe,
        hyperconnection_weight_IoTf2,
        hyperconnection_weight_ef2,
        hyperconnection_weight_ef1,
        hyperconnection_weight_f2f1,
        hyperconnection_weight_f2c,
        hyperconnection_weight_f1c,
    )

    # IoT node
    iot_output = Input(shape=(num_vars,))
    iot_skip_output = define_MLP_ResiliNet_architecture_IoT(iot_output, hidden_units)

    # failout definitions
    (
        edge_failout_lambda,
        fog2_failout_lambda,
        fog1_failout_lambda,
    ) = MLP_failout_definitions(failout_survival_setting)

    # edge node
    edge_output = define_MLP_ResiliNet_architecture_edge(
        iot_output, hidden_units, multiply_hyperconnection_weight_layer_IoTe
    )
    edge_output = edge_failout_lambda(edge_output)

    # fog node 2
    fog2_output = define_MLP_ResiliNet_architecture_fog2(
        iot_skip_output,
        edge_output,
        hidden_units,
        multiply_hyperconnection_weight_layer_IoTf2,
        multiply_hyperconnection_weight_layer_ef2,
    )
    fog2_output = fog2_failout_lambda(fog2_output)

    # fog node 1
    fog1_output = define_MLP_ResiliNet_architecture_fog1(
        edge_output,
        fog2_output,
        hidden_units,
        multiply_hyperconnection_weight_layer_ef1,
        multiply_hyperconnection_weight_layer_f2f1,
    )
    fog1_output = fog1_failout_lambda(fog1_output)

    # cloud node
    cloud_output = define_MLP_ResiliNet_architecture_cloud(
        fog2_output,
        fog1_output,
        hidden_units,
        num_classes,
        multiply_hyperconnection_weight_layer_f1c,
        multiply_hyperconnection_weight_layer_f2c,
    )

    model = Model(inputs=iot_output, outputs=cloud_output)
    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


def MLP_failout_definitions(failout_survival_setting):
    fog1_reliability = failout_survival_setting[0]
    fog2_reliability = failout_survival_setting[1]
    edge_reliability = failout_survival_setting[2]

    edge_failout_lambda = Failout(edge_reliability)
    fog2_failout_lambda = Failout(fog2_reliability)
    fog1_failout_lambda = Failout(fog1_reliability)
    return edge_failout_lambda, fog2_failout_lambda, fog1_failout_lambda


def define_MLP_ResiliNet_architecture_IoT(img_input, hidden_units):
    return define_MLP_DFG_architecture_IoT(img_input, hidden_units)


def define_MLP_ResiliNet_architecture_edge(
    iot_output, hidden_units, multiply_hyperconnection_weight_layer_IoTe=None
):
    return define_MLP_DFG_architecture_edge(
        iot_output, hidden_units, multiply_hyperconnection_weight_layer_IoTe
    )


def define_MLP_ResiliNet_architecture_fog2(
    iot_skip_output,
    edge_output,
    hidden_units,
    multiply_hyperconnection_weight_layer_IoTf2=None,
    multiply_hyperconnection_weight_layer_ef2=None,
):
    if (
        multiply_hyperconnection_weight_layer_IoTf2 == None
        or multiply_hyperconnection_weight_layer_ef2 == None
    ):
        fog2_input = Lambda(InputMux(ResiliNetPlus), name="node3_input")(
            [iot_skip_output, edge_output]
        )
    else:
        fog2_input = Lambda(InputMux(ResiliNetPlus), name="node3_input")(
            [
                multiply_hyperconnection_weight_layer_IoTf2(iot_skip_output),
                multiply_hyperconnection_weight_layer_ef2(edge_output),
            ]
        )
    fog2_output = define_MLP_architecture_fog2(fog2_input, hidden_units)
    return fog2_output


def define_MLP_ResiliNet_architecture_fog1(
    edge_skip_output,
    fog2_output,
    hidden_units,
    multiply_hyperconnection_weight_layer_ef1=None,
    multiply_hyperconnection_weight_layer_f2f1=None,
):
    if (
        multiply_hyperconnection_weight_layer_ef1 == None
        or multiply_hyperconnection_weight_layer_f2f1 == None
    ):
        fog1_input = Lambda(InputMux(ResiliNetPlus), name="node2_input")(
            [edge_skip_output, fog2_output]
        )
    else:
        fog1_input = Lambda(InputMux(ResiliNetPlus), name="node2_input")(
            [
                multiply_hyperconnection_weight_layer_ef1(edge_skip_output),
                multiply_hyperconnection_weight_layer_f2f1(fog2_output),
            ]
        )
    fog1_output = define_MLP_architecture_fog1(fog1_input, hidden_units)
    return fog1_output


def define_MLP_ResiliNet_architecture_cloud(
    fog2_skip_output,
    fog1_output,
    hidden_units,
    num_classes,
    multiply_hyperconnection_weight_layer_f1c=None,
    multiply_hyperconnection_weight_layer_f2c=None,
):
    if (
        multiply_hyperconnection_weight_layer_f1c == None
        or multiply_hyperconnection_weight_layer_f2c == None
    ):
        cloud_input = Lambda(InputMux(ResiliNetPlus), name="node1_input")(
            [fog2_skip_output, fog1_output]
        )
    else:
        cloud_input = Lambda(InputMux(ResiliNetPlus), name="node1_input")(
            [
                multiply_hyperconnection_weight_layer_f2c(fog2_skip_output),
                multiply_hyperconnection_weight_layer_f1c(fog1_output),
            ]
        )
    cloud_output = define_MLP_architecture_cloud(cloud_input, hidden_units, num_classes)
    return cloud_output
