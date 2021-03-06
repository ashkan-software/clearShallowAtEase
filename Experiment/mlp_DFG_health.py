import random

import keras.layers as layers
from Experiment.mlp_Vanilla_health import (
    define_MLP_architecture_cloud,
    define_MLP_architecture_edge,
    define_MLP_architecture_fog1,
    define_MLP_architecture_fog2,
)
from keras.layers import Activation, Dense, Input, Lambda
from keras.models import Model, Sequential


default_skip_hyperconnection_config = [1, 1, 1]


def define_DFG_MLP(
    num_vars,
    num_classes,
    hidden_units,
    reliability_setting=[
        1.0,
        1.0,
        1.0,
    ],  # reliability of a node between 0 and 1, [f1,f2,e1]
    skip_hyperconnection_config=default_skip_hyperconnection_config,  # binary representating if a skip hyperconnection is alive [f2,e1,g1]
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
    iot_skip_output = define_MLP_DFG_architecture_IoT(iot_output, hidden_units)

    # edge node
    edge_output = define_MLP_DFG_architecture_edge(
        iot_output, hidden_units, multiply_hyperconnection_weight_layer_IoTe
    )

    # fog node 2
    fog2_output = define_MLP_DFG_architecture_fog2(
        iot_skip_output,
        edge_output,
        hidden_units,
        multiply_hyperconnection_weight_layer_IoTf2,
        multiply_hyperconnection_weight_layer_ef2,
    )

    # fog node 1
    fog1_output = define_MLP_DFG_architecture_fog1(
        edge_output,
        fog2_output,
        hidden_units,
        multiply_hyperconnection_weight_layer_ef1,
        multiply_hyperconnection_weight_layer_f2f1,
    )

    # cloud node
    cloud_output = define_MLP_DFG_architecture_cloud(
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


def set_hyperconnection_weights(
    hyperconnection_weights_scheme, reliability_setting, skip_hyperconnection_config
):
    # weighted by 1
    if hyperconnection_weights_scheme == 1:
        hyperconnection_weight_IoTe = 1
        hyperconnection_weight_IoTf2 = 1
        hyperconnection_weight_ef2 = 1
        hyperconnection_weight_ef1 = 1
        hyperconnection_weight_f2f1 = 1
        hyperconnection_weight_f2c = 1
        hyperconnection_weight_f1c = 1
    # normalized reliability
    elif hyperconnection_weights_scheme == 2:
        hyperconnection_weight_IoTe = 1
        hyperconnection_weight_IoTf2 = 1 / (1 + reliability_setting[2])
        hyperconnection_weight_ef2 = reliability_setting[2] / (
            1 + reliability_setting[2]
        )
        hyperconnection_weight_ef1 = reliability_setting[2] / (
            reliability_setting[2] + reliability_setting[1]
        )
        hyperconnection_weight_f2f1 = reliability_setting[1] / (
            reliability_setting[2] + reliability_setting[1]
        )
        hyperconnection_weight_f2c = reliability_setting[1] / (
            reliability_setting[1] + reliability_setting[0]
        )
        hyperconnection_weight_f1c = reliability_setting[0] / (
            reliability_setting[1] + reliability_setting[0]
        )
    # reliability
    elif hyperconnection_weights_scheme == 3:
        hyperconnection_weight_IoTe = 1
        hyperconnection_weight_IoTf2 = 1
        hyperconnection_weight_ef2 = reliability_setting[2]
        hyperconnection_weight_ef1 = reliability_setting[2]
        hyperconnection_weight_f2f1 = reliability_setting[1]
        hyperconnection_weight_f2c = reliability_setting[1]
        hyperconnection_weight_f1c = reliability_setting[0]
    # randomly weighted between 0 and 1
    elif hyperconnection_weights_scheme == 4:
        hyperconnection_weight_IoTe = random.uniform(0, 1)
        hyperconnection_weight_IoTf2 = random.uniform(0, 1)
        hyperconnection_weight_ef2 = random.uniform(0, 1)
        hyperconnection_weight_ef1 = random.uniform(0, 1)
        hyperconnection_weight_f2f1 = random.uniform(0, 1)
        hyperconnection_weight_f2c = random.uniform(0, 1)
        hyperconnection_weight_f1c = random.uniform(0, 1)
    # randomly weighted between 0 and 10
    elif hyperconnection_weights_scheme == 5:
        hyperconnection_weight_IoTe = random.uniform(0, 10)
        hyperconnection_weight_IoTf2 = random.uniform(0, 10)
        hyperconnection_weight_ef2 = random.uniform(0, 10)
        hyperconnection_weight_ef1 = random.uniform(0, 10)
        hyperconnection_weight_f2f1 = random.uniform(0, 10)
        hyperconnection_weight_f2c = random.uniform(0, 10)
        hyperconnection_weight_f1c = random.uniform(0, 10)
    else:
        raise ValueError("Incorrect scheme value")
    (
        hyperconnection_weight_IoTf2,
        hyperconnection_weight_ef1,
        hyperconnection_weight_f2c,
    ) = remove_skip_hyperconnection_for_sensitvity_experiment(
        skip_hyperconnection_config,
        hyperconnection_weight_IoTf2,
        hyperconnection_weight_ef1,
        hyperconnection_weight_f2c,
    )
    return (
        hyperconnection_weight_IoTe,
        hyperconnection_weight_IoTf2,
        hyperconnection_weight_ef2,
        hyperconnection_weight_ef1,
        hyperconnection_weight_f2f1,
        hyperconnection_weight_f2c,
        hyperconnection_weight_f1c,
    )


def remove_skip_hyperconnection_for_sensitvity_experiment(
    skip_hyperconnection_config,
    hyperconnection_weight_IoTf2,
    hyperconnection_weight_ef1,
    hyperconnection_weight_f2c,
):
    # take away the skip hyperconnection if the value in hyperconnections array is 0
    # from fog node 2 to cloud node
    if skip_hyperconnection_config[0] == 0:
        hyperconnection_weight_f2c = 0
    # from edge node to fog node 1
    if skip_hyperconnection_config[1] == 0:
        hyperconnection_weight_ef1 = 0
    # from IoT node to fog node 2
    if skip_hyperconnection_config[2] == 0:
        hyperconnection_weight_IoTf2 = 0
    return (
        hyperconnection_weight_IoTf2,
        hyperconnection_weight_ef1,
        hyperconnection_weight_f2c,
    )


def define_hyperconnection_weight_lambda_layers(
    hyperconnection_weight_IoTe,
    hyperconnection_weight_IoTf2,
    hyperconnection_weight_ef2,
    hyperconnection_weight_ef1,
    hyperconnection_weight_f2f1,
    hyperconnection_weight_f2c,
    hyperconnection_weight_f1c,
):
    # define lambdas for multiplying node weights by connection weight
    multiply_hyperconnection_weight_layer_IoTe = Lambda(
        (lambda x: x * hyperconnection_weight_IoTe), name="hyperconnection_weight_IoTe"
    )
    multiply_hyperconnection_weight_layer_IoTf2 = Lambda(
        (lambda x: x * hyperconnection_weight_IoTf2),
        name="hyperconnection_weight_IoTf2",
    )
    multiply_hyperconnection_weight_layer_ef2 = Lambda(
        (lambda x: x * hyperconnection_weight_ef2), name="hyperconnection_weight_ef2"
    )
    multiply_hyperconnection_weight_layer_ef1 = Lambda(
        (lambda x: x * hyperconnection_weight_ef1), name="hyperconnection_weight_ef1"
    )
    multiply_hyperconnection_weight_layer_f2f1 = Lambda(
        (lambda x: x * hyperconnection_weight_f2f1), name="hyperconnection_weight_f2f1"
    )
    multiply_hyperconnection_weight_layer_f2c = Lambda(
        (lambda x: x * hyperconnection_weight_f2c), name="hyperconnection_weight_f2c"
    )
    multiply_hyperconnection_weight_layer_f1c = Lambda(
        (lambda x: x * hyperconnection_weight_f1c), name="hyperconnection_weight_f1c"
    )
    return (
        multiply_hyperconnection_weight_layer_IoTe,
        multiply_hyperconnection_weight_layer_IoTf2,
        multiply_hyperconnection_weight_layer_ef2,
        multiply_hyperconnection_weight_layer_ef1,
        multiply_hyperconnection_weight_layer_f2f1,
        multiply_hyperconnection_weight_layer_f2c,
        multiply_hyperconnection_weight_layer_f1c,
    )


def define_MLP_DFG_architecture_IoT(img_input, hidden_units):
    # use a linear Dense layer to transform input into the shape needed for the network
    iot_output = Dense(units=hidden_units, name="skip_iotfog2", activation="linear")(
        img_input
    )
    return iot_output


def define_MLP_DFG_architecture_edge(
    iot_output, hidden_units, multiply_hyperconnection_weight_layer_IoTe=None
):
    if multiply_hyperconnection_weight_layer_IoTe != None:
        iot_output = multiply_hyperconnection_weight_layer_IoTe(iot_output)
    edge_output = define_MLP_architecture_edge(iot_output, hidden_units)
    return edge_output


def define_MLP_DFG_architecture_fog2(
    iot_output,
    edge_output,
    hidden_units,
    multiply_hyperconnection_weight_layer_IoTf2=None,
    multiply_hyperconnection_weight_layer_ef2=None,
):
    if (
        multiply_hyperconnection_weight_layer_IoTf2 == None
        or multiply_hyperconnection_weight_layer_ef2 == None
    ):
        fog2_input = layers.add([edge_output, iot_output], name="node3_input")
    else:
        fog2_input = layers.add(
            [
                multiply_hyperconnection_weight_layer_ef2(edge_output),
                multiply_hyperconnection_weight_layer_IoTf2(iot_output),
            ],
            name="node3_input",
        )
    fog2_output = define_MLP_architecture_fog2(fog2_input, hidden_units)
    return fog2_output


def define_MLP_DFG_architecture_fog1(
    edge_output,
    fog2_output,
    hidden_units,
    multiply_hyperconnection_weight_layer_ef1=None,
    multiply_hyperconnection_weight_layer_f2f1=None,
):
    if (
        multiply_hyperconnection_weight_layer_ef1 == None
        or multiply_hyperconnection_weight_layer_f2f1 == None
    ):
        fog1_input = layers.add([edge_output, fog2_output], name="node2_input")
    else:
        fog1_input = layers.add(
            [
                multiply_hyperconnection_weight_layer_ef1(edge_output),
                multiply_hyperconnection_weight_layer_f2f1(fog2_output),
            ],
            name="node2_input",
        )
    fog1_output = define_MLP_architecture_fog1(fog1_input, hidden_units)
    return fog1_output


def define_MLP_DFG_architecture_cloud(
    fog2_output,
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
        cloud_input = layers.add([fog1_output, fog2_output], name="node1_input")
    else:
        cloud_input = layers.add(
            [
                multiply_hyperconnection_weight_layer_f1c(fog1_output),
                multiply_hyperconnection_weight_layer_f2c(fog2_output),
            ],
            name="node1_input",
        )
    cloud_output = define_MLP_architecture_cloud(cloud_input, hidden_units, num_classes)
    return cloud_output
