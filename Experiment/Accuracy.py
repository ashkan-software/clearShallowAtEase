import sys

import keras.backend as K
import numpy as np
from Experiment.cnn_Vanilla_MobileNet import (
    PARTITION_SETING as PARTITION_SETING_MobileNet,
)
from Experiment.cnn_Vanilla_ResNet import PARTITION_SETING as PARTITION_SETING_ResNet
from Experiment.common import convert_binary_to_list
from Experiment.evaluation import predict


model_accuracy_dict = dict()


class accuracy:
    experiment_name = ""

    def __init__(self, exp_name):
        self.experiment_name = exp_name

    def fail_node(self, model, node_failure_combination):
        """fails node(s) by making the specified node(s) output 0

        Args:
            model (Model): Keras model to have nodes failed
            node_failure_combination (list): bit list that corresponds
                to the node failure combination, 1 in the list represents
                to alive and 0 corresponds to dead. they are ordered from
                top to down, left to right
        Returns:
            returns a boolean whether the model failed was a cnn or not
        """

        def set_weights_zero_MLP(model, layers, index):
            layer_name = layers[index]
            layer = model.get_layer(name=layer_name)
            layer_weights = layer.get_weights()
            # make new weights for the connections
            new_weights = np.zeros(layer_weights[0].shape)
            # make new weights for biases
            new_bias_weights = np.zeros(layer_weights[1].shape)
            layer.set_weights([new_weights, new_bias_weights])

        def set_weights_zero_CNN(model, layers, index):
            layer_name = layers[index]
            layer = model.get_layer(name=layer_name)
            layer_weights = layer.get_weights()
            # make new weights for the connections
            new_weights = np.zeros(layer_weights[0].shape)
            # make new weights for biases

            # kill batch normalizations
            if "bn" in layer_name:
                layer.set_weights([new_weights, new_weights, new_weights, new_weights])
            else:
                new_bias_weights = np.zeros(layer_weights[1].shape)
                layer.set_weights([new_weights, new_bias_weights])

        def set_weights_zero_CNN_ResNet(model, layers_edge, layers_fog, index):
            if index == 0:  # fog node fails
                fail_layers = layers_fog
            elif index == 1:  # edge node fails
                fail_layers = layers_edge
            else:
                print("Error! wrong index for node failure:", index)
                sys.exit()
            for i, _ in enumerate(fail_layers):
                set_weights_zero_CNN(model, fail_layers, i)

        # input is image
        if self.experiment_name == "Camera":
            layers = [
                "fog1_output_layer",
                "fog2_output_layer",
                "fog3_output_layer",
                "fog4_output_layer",
                "edge1_output_layer",
                "edge2_output_layer",
                "edge3_output_layer",
                "edge4_output_layer",
            ]
            for index, node in enumerate(node_failure_combination):
                if node == 0:  # if dead
                    set_weights_zero_MLP(model, layers, index)

        elif self.experiment_name == "CIFAR" or self.experiment_name == "Imagenet":
            if PARTITION_SETING_MobileNet == 1:
                layers = ["conv_pw_8_bn", "conv_pw_3_bn"]
            else:  # PARTITION_SETING_MobileNet == 2
                layers = ["conv_pw_7_bn", "conv_pw_3_bn"]
            for index, node in enumerate(node_failure_combination):
                if node == 0:  # dead
                    set_weights_zero_CNN(model, layers, index)
        elif self.experiment_name == "ResNet":
            if PARTITION_SETING_ResNet == 1:
                layers_edge = [
                    "conv_6",
                    "conv_3",
                    "skip_bn_4",
                ]  # skip_bn_4 is the last one in the "recursion"
                layers_fog = [
                    "conv_11",
                    "conv_8",
                    "skip_bn_9",
                ]  # skip_bn_9 is the last one in the "recursion"
            else:  # PARTITION_SETING_ResNet == 2
                layers_edge = [
                    "conv_11",
                    "conv_8",
                    "skip_bn_9",
                ]  # skip_bn_9 is the last one in the "recursion"
                layers_fog = [
                    "conv_16",
                    "conv_13",
                    "skip_bn_14",
                ]  # skip_bn_14 is the last one in the "recursion"
            for index, node in enumerate(node_failure_combination):
                if node == 0:  # dead
                    set_weights_zero_CNN_ResNet(model, layers_edge, layers_fog, index)
        elif self.experiment_name == "Health":
            layers = ["fog1_output_layer", "fog2_output_layer", "edge_output_layer"]
            for index, node in enumerate(node_failure_combination):
                # node failed
                if node == 0:
                    set_weights_zero_MLP(model, layers, index)
        elif self.experiment_name is not "Imagenet":
            print("Error! Please specify the correct experiment name")
            sys.exit()

    def iterate_failure_combinations(
        self,
        reliability_setting,
        num_nodes,
        model,
        no_information_flow_map,
        output_list,
        training_labels=None,
        test_data=None,
        test_labels=None,
        test_generator=None,
        num_test_examples=None,
    ):
        """Runs through all node failure combinations and calculates
            the accuracy (and weight) of that particular node failure combination

        Args:
            reliability_setting (list): List of the reliability of all nodes,
                ordered from fog to edge node
            num_nodes (int): number of physical nodes
            model (Model): Keras model
            no_information_flow_map (dictionary): a dictionary that maps from a
                certain node failure combination to a boolean, showing if that
                node failure combination has an accessible path of information
                to the cloud.
            output_list (list): list that contains string output of the experiment
            training_labels (numpy array): 1D array that corresponds to each row in
                the training data with a class label, used for calculating train
                class distributio
            test_data (numpy array): 2D array that contains the test data, assumes
                that each column is a variable and that each row is a test example
            test_labels (numpy array): 1D array that corresponds to each row in
                the test data with a class label
            test_generator (generator): a generator for test data (for imagenet)
            num_test_examples (int): number of test examples (for imagenet)
        Returns:
            return accuracy and weight of each node failure combination
        """
        weight_list = []
        need_calculating_accuracy = False
        if model in model_accuracy_dict:  # if the accuracy for this model is calculated
            accuracy_list = model_accuracy_dict[model]
        else:
            accuracy_list = []
            need_calculating_accuracy = True

        output_list.append(
            "Calculating accuracy for reliability setting "
            + str(reliability_setting)
            + "\n"
        )
        print(
            "Calculating accuracy for reliability setting " + str(reliability_setting)
        )
        max_node_failures = 2 ** num_nodes
        for i in range(max_node_failures):
            node_failure_combination = convert_binary_to_list(i, num_nodes)
            # print(node_failure_combination)
            if need_calculating_accuracy:
                # saves a copy of the original model so it does not change during failures
                no_information_flow = no_information_flow_map[
                    tuple(node_failure_combination)
                ]
                if not no_information_flow:
                    old_weights = model.get_weights()
                    self.fail_node(model, node_failure_combination)
                output_list.append(str(node_failure_combination))
                if self.experiment_name == "Imagenet":
                    if no_information_flow:
                        accuracy = 0.001
                    else:
                        accuracy = model.evaluate_generator(
                            test_generator,
                            steps=num_test_examples / test_generator.batch_size,
                        )[1]
                else:
                    accuracy, _ = predict(
                        model,
                        no_information_flow,
                        training_labels,
                        test_data,
                        test_labels,
                        self.experiment_name,
                    )
                accuracy_list.append(accuracy)
                if not no_information_flow:
                    model.set_weights(
                        old_weights
                    )  # change the changed weights to the original weights
            weight = calc_weight_probability(
                reliability_setting, node_failure_combination
            )
            weight_list.append(weight)
        print("Acc List: " + str(accuracy_list))
        output_list.append("Acc List: " + str(accuracy_list) + "\n")

        if need_calculating_accuracy:
            model_accuracy_dict[
                model
            ] = accuracy_list  # add the accuracy_list to the dictionary
        return accuracy_list, weight_list

    def calc_expected_accuracy(
        self,
        model,
        no_information_flow_map,
        reliability_setting,
        output_list,
        training_labels=None,
        test_data=None,
        test_labels=None,
        test_generator=None,
        num_test_examples=None,
    ):
        """Calculates the expected accuracy of the model under a reliability setting

        Args:
            model (Model): Keras model
            no_information_flow_map (dictionary): a dictionary that maps from a
                certain node failure combination to a boolean, showing if that
                node failure combination has an accessible path of information
                to the cloud.
            reliability_setting (list): List of the reliability rate of all nodes
            output_list (list): list that contains string output of the experiment
            training_labels (numpy array): 1D array that corresponds to each row in
                the training data with a class label, used for calculating train
                class distributio
            test_data (numpy array): 2D array that contains the test data, assumes
                that each column is a variable and that each row is a test example
            test_labels (numpy array): 1D array that corresponds to each row in the
                test data with a class label
            test_generator (generator): a generator for test data (for imagenet)
            num_test_examples (int): number of test examples (for imagenet)
        Returns:
            return weighted accuracy
        """
        num_nodes = len(reliability_setting)
        accuracy_list, weight_list = self.iterate_failure_combinations(
            reliability_setting,
            num_nodes,
            model,
            no_information_flow_map,
            output_list,
            training_labels,
            test_data,
            test_labels,
            test_generator,
            num_test_examples,
        )
        weight_list = normalize(weight_list)
        avg_acc = calc_weighted_average(accuracy_list, weight_list)
        output_list.append("Average accuracy: " + str(avg_acc) + "\n")
        print("Average accuracy:", avg_acc)
        return avg_acc


def calc_weighted_average(valueList, weight_list):
    """calculates weighted average

    Args:
        valueList (list): list of all the values
        weight_list (list): list of all weights (probabilities) of those values

    Returns:
        return the weighted average
    """
    average = 0
    for i in range(len(valueList)):
        average += valueList[i] * weight_list[i]
    return average


def calc_weight_probability(reliability_setting, node_failure_combination):
    """calculates the weight (probability) of each combination of node failures

    Args:
        reliability_setting (list): list of probabilities
        node_failure_combination (list): bit list that corresponds
            to the node failure combination, 1 in the list represents
            to alive and 0 corresponds to dead. they are ordered from
            top to down, left to right
    Returns:
        return probability of a particular node failure combination
    """
    weight = 1
    for i in range(len(node_failure_combination)):
        if node_failure_combination[i] == 1:  # if it survives
            weight = weight * reliability_setting[i]
        else:  # if it fails
            weight = weight * (1 - reliability_setting[i])
    return weight


def calc_num_survived_nodes(number):
    """calculates the number of survived physical nodes by counting ones
        in a bit string

    Args:
        number (int): number to be converted to binary

    Returns:
        return number of survived nodes
    """
    # convert given number into binary
    # output will be like bin(11)=0b1101
    binary = bin(number)
    # now separate out all 1's from binary string
    # we need to skip starting two characters
    # of binary string i.e; 0b
    setBits = [ones for ones in binary[2:] if ones == "1"]
    return len(setBits)


def normalize(weights):
    """Normalizes the elements of a list, so that they sum to 1

    Args:
       weights(list): list of all the probability weights

    Returns:
        return normalized lost of probability weights
    """
    sumWeights = sum(weights)
    normalized = [(x / sumWeights) for x in weights]
    return normalized
