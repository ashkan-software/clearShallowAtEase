
from Experiment.classification import predict
import keras.backend as K
import numpy as np
import sys
from Experiment.common_exp_methods import convertBinaryToList
from Experiment.cnn_Vanilla_ResNet import PARTITION_SETING

modelAccuracyDict = dict()

class accuracy:
    experiment_name = ""
    def __init__(self,exp_name):
        self.experiment_name = exp_name

    def fail_node(self, model,node_failure_combination):
        """fails node(s) by making the specified node(s) output 0
        ### Arguments
            model (Model): Keras model to have nodes failed
            node_failure_combination (list): bit list that corresponds to the node failure combination, 1 in the list represents to alive and 0 corresponds to dead. they are ordered from top to down, left to right (like from f1,f2,...,e1,e2,...)
        ### Returns
            return a boolean whether the model failed was a cnn or not
        """
        def set_weights_zero_MLP(model, layers, index):
            layer_name = layers[index]
            layer = model.get_layer(name=layer_name)
            layer_weights = layer.get_weights()
            # make new weights for the connections
            new_weights = np.zeros(layer_weights[0].shape)
            # make new weights for biases
            new_bias_weights = np.zeros(layer_weights[1].shape)
            layer.set_weights([new_weights,new_bias_weights])

        def set_weights_zero_CNN(model, layers, index):
            layer_name = layers[index]
            layer = model.get_layer(name=layer_name)
            layer_weights = layer.get_weights()
            # make new weights for the connections
            new_weights = np.zeros(layer_weights[0].shape)
             # make new weights for biases
            new_bias_weights = np.zeros(layer_weights[1].shape)
            layer.set_weights([new_weights,new_bias_weights])

        def set_weights_zero_CNN_ResNet(model, layers_edge, layers_fog, index):
            if index == 0: # fog node fails
                fail_layers = layers_fog
            elif index == 1: # edge node fails
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
                "edge4_output_layer"
                ]
            for index, node in enumerate(node_failure_combination):
                if node == 0: # if dead
                    set_weights_zero_MLP(model, layers, index)
            
        elif self.experiment_name == "CIFAR" or self.experiment_name == "Imagenet": 
            layers = ["conv_pw_8","conv_pw_3"]
            for index,node in enumerate(node_failure_combination):
                if node == 0: # dead
                    set_weights_zero_CNN(model, layers, index)
        elif self.experiment_name == "ResNet":
            if PARTITION_SETING == 1:
                layers_edge = ["conv_6","conv_3","skip_conv_4"] # skip_conv_4 is the last one in the "recursion"
                layers_fog = ["conv_11","conv_8","skip_conv_9"] # skip_conv_9 is the last one in the "recursion"
            else: # PARTITION_SETING == 2
                layers_edge = ["conv_11","conv_8","skip_conv_9"] # skip_conv_9 is the last one in the "recursion"
                layers_fog = ["conv_16","conv_13","skip_conv_14"] # skip_conv_14 is the last one in the "recursion"
            for index,node in enumerate(node_failure_combination):
                if node == 0: # dead
                    set_weights_zero_CNN_ResNet(model, layers_edge, layers_fog, index)
        elif self.experiment_name == "Health":              
            layers = ["fog1_output_layer","fog2_output_layer","edge_output_layer"]
            for index,node in enumerate(node_failure_combination):
                # node failed
                if node == 0:
                    set_weights_zero_MLP(model, layers, index)
        elif self.experiment_name is not "Imagenet":
            print("Error! Please specify the correct experiment name")
            sys.exit()

    def iterateAllFailureCombinationsCalcAccuracy(self,
                                                reliability_setting,
                                                numNodes,
                                                model,
                                                no_information_flow_map,
                                                output_list,
                                                training_labels = None,
                                                test_data = None,
                                                test_labels = None,
                                                test_generator = None, # for imageNet
                                                num_test_examples = None # for imageNet
                                                ):
        """runs through all node failure combinations and calculates the accuracy (and weight) of that particular node failure combination
        ### Arguments
            reliability_setting (list): List of the reliability of all nodes, ordered from edge to fog node
            numNodes (int): number of physical nodes
            model (Model): Keras model
            output_list (list): list that contains string output of the experiment
            train_labels (numpy array): 1D array that corresponds to each row in the training data with a class label, used for calculating train class distributio
            test_data (numpy array): 2D array that contains the test data, assumes that each column is a variable and that each row is a test example
            test_labels (numpy array): 1D array that corresponds to each row in the test data with a class label
        ### Returns
            return accuracy and weight of each node failure combination
        """ 
        weightList = []
        needToGetModelAccuracy = False
        if model in modelAccuracyDict: # if the accuracy for this model is calculated
            accuracyList = modelAccuracyDict[model]
        else:
            accuracyList = []
            needToGetModelAccuracy = True

        output_list.append('Calculating accuracy for reliability setting ' + str(reliability_setting) + '\n')
        print("Calculating accuracy for reliability setting "+ str(reliability_setting))
        maxNumNodeFailure = 2 ** numNodes
        for i in range(maxNumNodeFailure):
            node_failure_combination = convertBinaryToList(i, numNodes)
            # print(node_failure_combination)
            if needToGetModelAccuracy:
                # saves a copy of the original model so it does not change during failures 
                no_information_flow = no_information_flow_map[tuple(node_failure_combination)]
                if not no_information_flow:
                    old_weights = model.get_weights()
                    self.fail_node(model,node_failure_combination)
                output_list.append(str(node_failure_combination))
                if self.experiment_name == "Imagenet":
                    if no_information_flow:
                        accuracy = 0.001
                    else:
                        accuracy = model.evaluate_generator(test_generator, steps = num_test_examples / test_generator.batch_size)[1]
                else: 
                    accuracy,_ = predict(model,no_information_flow,training_labels,test_data,test_labels, self.experiment_name)
                accuracyList.append(accuracy)
                if not no_information_flow:
                    model.set_weights(old_weights) # change the changed weights to the original weights
            weight = calcWeightProbability(reliability_setting, node_failure_combination)
            weightList.append(weight)
        print("Acc List: " + str(accuracyList))
        output_list.append("Acc List: " + str(accuracyList) + '\n')
        
        if needToGetModelAccuracy:
            modelAccuracyDict[model] = accuracyList # add the accuracyList to the dictionary
        return accuracyList, weightList
 
    def calculateExpectedAccuracy(self,
                                model,
                                no_information_flow_map,
                                reliability_setting,
                                output_list,
                                training_labels = None,
                                test_data = None,
                                test_labels = None,
                                test_generator = None, # for imageNet
                                num_test_examples = None # for imageNet
                                ):
        """Calculates the expected accuracy of the model under certain reliability setting
        ### Arguments
            model (Model): Keras model
            reliability_setting (list): List of the reliability rate of all nodes
            output_list (list): list that contains string output of the experiment
            training_labels (numpy array): 1D array that corresponds to each row in the training data with a class label, used for calculating train class distributio
            test_data (numpy array): 2D array that contains the test data, assumes that each column is a variable and that each row is a test example
            test_labels (numpy array): 1D array that corresponds to each row in the test data with a class label
        ### Returns
            return weighted accuracy 
        """  
        K.set_learning_phase(0)
        numNodes = len(reliability_setting)
        accuracyList, weightList = self.iterateAllFailureCombinationsCalcAccuracy(reliability_setting,numNodes, model,no_information_flow_map,output_list,training_labels,test_data,test_labels, test_generator, num_test_examples)
        weightList = normalize(weightList)
        avg_acc = calcWeightedAverage(accuracyList, weightList)
        # output_list.append('Times we had no information flow: ' + str(no_information_flow_count) + '\n')
        output_list.append('Average Accuracy: ' + str(avg_acc) + '\n')
        # print('Times we had no information flow: ',str(no_information_flow_count))
        print("Average Accuracy:", avg_acc)
        return avg_acc

def calcWeightedAverage(valueList, weightList):
    """calculates weighted average 
    ### Arguments
        valueList (list): list of all the values
        weightList (list): list of all weights (probabilities) of those values
    ### Returns
        return weighted average 
    """  
    average = 0
    for i in range(len(valueList)):
        average += valueList[i] * weightList[i]
    return average
        
def calcWeightProbability(reliability_setting, node_failure_combination):
    """calculates the weight (probability) of each combination of node failures
    ### Arguments
        reliability_setting (list): list of probabilities
    ### Returns
        return probability of a particular node failure combination
    """  
    weight = 1
    for i in range(len(node_failure_combination)):
        if (node_failure_combination[i] == 1): # if it survives
            weight = weight * reliability_setting[i]
        else: # if it fails
            weight = weight * (1 - reliability_setting[i])
    return weight
    

def calcNumSurvivedNodes(number):
    """calculates the number of survived physical nodes by counting ones in a bit string
    ### Arguments
        number (int): number to be converted to binary
    ### Returns
        return number of survived nodes
    """  
    # convert given number into binary
    # output will be like bin(11)=0b1101
    binary = bin(number)
    # now separate out all 1's from binary string
    # we need to skip starting two characters
    # of binary string i.e; 0b
    setBits = [ones for ones in binary[2:] if ones=='1']
    return len(setBits)

def normalize(weights):
    """Normalizes the elements of a list, so that they sum to 1
    ### Arguments
       weights(list): list of all the probability weights
    ### Returns
        return normalized lost of probability weights
    """  
    sumWeights = sum(weights)
    normalized = [(x/sumWeights) for x in weights]
    return normalized