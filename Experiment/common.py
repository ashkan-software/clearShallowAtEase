import copy
import os

import keras
import numpy as np
import tensorflow as tf
from Experiment.graph import (
    create_graph_CNN,
    create_graph_MLP_camera,
    create_graph_MLP_health,
    fail_node_graph,
    identify_no_information_flow_graph,
)
from keras import optimizers
from keras.utils import multi_gpu_model


def make_results_folder():
    # makes folder for results and models (if they don't exist)
    if not os.path.exists("results/"):
        os.mkdir("results/")
    if not os.path.exists("models"):
        os.mkdir("models/")


def save_output(output_name, output_list):
    # write experiments output to file
    with open(output_name, "w") as file:
        file.writelines(output_list)
        file.flush()
        os.fsync(file)


def convert_to_string(reliability_settings):
    # convert reliability settings into strings so it can be used in the dictionary as keys
    no_failure = str(reliability_settings[0])
    normal = str(reliability_settings[1])
    poor = str(reliability_settings[2])
    hazardous = str(reliability_settings[3])
    return no_failure, normal, poor, hazardous


def make_output_dictionary_average_accuracy(reliability_settings, num_iterations):
    no_failure, normal, poor, hazardous = convert_to_string(reliability_settings)

    # dictionary to store all the results
    output = {
        "ResiliNet": {
            hazardous: [0] * num_iterations,
            poor: [0] * num_iterations,
            normal: [0] * num_iterations,
            no_failure: [0] * num_iterations,
        },
        "DFG": {
            hazardous: [0] * num_iterations,
            poor: [0] * num_iterations,
            normal: [0] * num_iterations,
            no_failure: [0] * num_iterations,
        },
        "Vanilla": {
            hazardous: [0] * num_iterations,
            poor: [0] * num_iterations,
            normal: [0] * num_iterations,
            no_failure: [0] * num_iterations,
        },
    }
    return output


def make_output_dictionary_hyperconnection_weight(
    model_name, reliability_settings, num_iterations
):
    no_failure, normal, poor, hazardous = convert_to_string(reliability_settings)

    # define weight schemes for hyperconnections
    one_weight_scheme = 1  # weighted by 1
    normalized_reliability_weight_scheme = 2  # normalized reliability
    reliability_weight_scheme = 3  # reliability
    random_weight_scheme = 4  # randomly weighted between 0 and 1
    random_weight_scheme2 = 5  # randomly weighted between 0 and 10

    weight_schemes = [
        one_weight_scheme,
        normalized_reliability_weight_scheme,
        reliability_weight_scheme,
        random_weight_scheme,
        random_weight_scheme2,
    ]

    # dictionary to store all the results
    output = {
        model_name: {
            one_weight_scheme: {
                no_failure: [0] * num_iterations,
                hazardous: [0] * num_iterations,
                poor: [0] * num_iterations,
                normal: [0] * num_iterations,
            },
            normalized_reliability_weight_scheme: {
                no_failure: [0] * num_iterations,
                hazardous: [0] * num_iterations,
                poor: [0] * num_iterations,
                normal: [0] * num_iterations,
            },
            reliability_weight_scheme: {
                no_failure: [0] * num_iterations,
                hazardous: [0] * num_iterations,
                poor: [0] * num_iterations,
                normal: [0] * num_iterations,
            },
            random_weight_scheme: {
                no_failure: [0] * num_iterations,
                hazardous: [0] * num_iterations,
                poor: [0] * num_iterations,
                normal: [0] * num_iterations,
            },
            random_weight_scheme2: {
                no_failure: [0] * num_iterations,
                hazardous: [0] * num_iterations,
                poor: [0] * num_iterations,
                normal: [0] * num_iterations,
            },
        }
    }
    return output, weight_schemes


def make_output_dictionary_failout_rate(
    failout_survival_rates, reliability_settings, num_iterations
):
    no_failure, normal, poor, hazardous = convert_to_string(reliability_settings)

    # dictionary to store all the results
    output = {}
    for failout_survival_rate in failout_survival_rates:
        output[str(failout_survival_rate)] = {
            hazardous: [0] * num_iterations,
            poor: [0] * num_iterations,
            normal: [0] * num_iterations,
            no_failure: [0] * num_iterations,
        }
    output["Variable Failout 1x"] = {
        hazardous: [0] * num_iterations,
        poor: [0] * num_iterations,
        normal: [0] * num_iterations,
        no_failure: [0] * num_iterations,
    }

    return output


def average(list):
    """function to return average of a list 

    Args:
        list (list): list of numbers
    
    Returns:
        return average of list
    """
    if len(list) == 0:
        return 0
    else:
        return sum(list) / len(list)


def compile_keras_parallel_model(input, cloud_output, num_gpus, name="ANRL_mobilenet"):
    # Create model.
    with tf.device("/cpu:0"):
        model = keras.Model(input, cloud_output, name=name)

    parallel_model = ""
    if num_gpus > 1:
        parallel_model = multi_gpu_model(model, gpus=num_gpus)
        # sgd_optimizer = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        adam_optimizer = optimizers.Adam(lr=0.001)
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=adam_optimizer,
            metrics=["accuracy"],
        )
        parallel_model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )
    else:
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )
    return model, parallel_model


def make_no_information_flow_map(exp, skip_hyperconnection_config=None):
    if exp == "CIFAR/Imagenet" or exp == "ResNet":
        graph = create_graph_CNN(skip_hyperconnection_config)
        num_nodes = 2
    if exp == "Health":
        graph = create_graph_MLP_health(skip_hyperconnection_config)
        num_nodes = 3
    if exp == "Camera":
        graph = create_graph_MLP_camera(skip_hyperconnection_config)
        num_nodes = 8
    max_node_failures = 2 ** num_nodes
    no_information_flow_map = {}  # make a dictionary
    for i in range(max_node_failures):
        node_failure_combination = convert_binary_to_list(i, num_nodes)
        graph_copy = copy.deepcopy(graph)  # make a new copy of the graph
        fail_node_graph(graph_copy, node_failure_combination, exp)
        no_information_flow_map[
            tuple(node_failure_combination)
        ] = identify_no_information_flow_graph(graph_copy, exp)
        del graph_copy
    return no_information_flow_map


def convert_binary_to_list(number, numBits):
    """converts a number (e.g. 128) to its binary representation in a list. It converts number 128 to [1,0,0,0,0,0,0,0]    
    
    Args:
        number (int): number to be converted to binary
        numBits (int): number of maximum bits 
    
    Returns:
        return binary number to a list representing the binary number
    """
    # convert given number into binary
    # output will be like bin(11)=0b1101
    binary = bin(number)
    lst = [bits for bits in binary[2:]]
    # pad '0's to the begging of the list, if its length is not 'numBits'
    for _ in range(max(0, numBits - len(lst))):
        lst.insert(0, "0")
    lst_integer = [int(num) for num in lst]  # converts ["0","1","0"] to [0,1,0]
    return lst_integer
