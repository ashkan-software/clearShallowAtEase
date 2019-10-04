
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


from Experiment.common_exp_methods import fail_node
from Experiment.classification import predict
import keras.backend as K

modelAccuracyDict = dict()

def iterateAllFailureCombinationsCalcAccuracy(reliability_setting,
                                            numNodes,
                                            model,
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

    if training_labels is None or test_data is None or test_labels is None:
        isImageNet = True
    else:
        isImageNet = False
    
    output_list.append('Calculating accuracy for reliability setting ' + str(reliability_setting) + '\n')
    print("Calculating accuracy for reliability setting "+ str(reliability_setting))
    maxNumNodeFailure = 2 ** numNodes
    for i in range(maxNumNodeFailure):
        node_failure_combination = convertBinaryToList(i, numNodes)
        # print(node_failure_combination)
        if needToGetModelAccuracy:
            # saves a copy of the original model so it does not change during failures 
            old_weights = model.get_weights()
            is_cnn = fail_node(model,node_failure_combination)
            output_list.append(str(node_failure_combination))
            if isImageNet:
                accuracy = model.evaluate_generator(test_generator, steps = num_test_examples / test_generator.batch_size)[1]  
            else: 
                accuracy,_ = predict(model,training_labels,test_data,test_labels, is_cnn)
            accuracyList.append(accuracy)
            # change the changed weights to the original weights
            model.set_weights(old_weights)
        weight = calcWeightProbability(reliability_setting, node_failure_combination)
        weightList.append(weight)
    print("Acc List: " + str(accuracyList))
    output_list.append("Acc List: " + str(accuracyList) + '\n')
    
    if needToGetModelAccuracy:
        modelAccuracyDict[model] = accuracyList # add the accuracyList to the dictionary
    return accuracyList, weightList

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

def convertBinaryToList(number, numBits):
    """converts a number (e.g. 128) to its binary representation in a list. It converts number 128 to [1,0,0,0,0,0,0,0]    
    ### Arguments
        number (int): number to be converted to binary
        numBits (int): number of maximum bits 
    ### Returns
        return binary number to a list representing the binary number
    """  
    # convert given number into binary
    # output will be like bin(11)=0b1101
    binary = bin(number)
    lst = [bits for bits in binary[2:]]
    # pad '0's to the begging of the list, if its length is not 'numBits'
    for _ in range(max(0,numBits - len(lst))):
        lst.insert(0,'0')
    lst_integer = [int(num) for num in lst] # converts ["0","1","0"] to [0,1,0]
    return lst_integer

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
 
def calculateExpectedAccuracy(model,
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
    accuracyList, weightList = iterateAllFailureCombinationsCalcAccuracy(reliability_setting,numNodes, model,output_list,training_labels,test_data,test_labels, test_generator, num_test_examples)
    weightList = normalize(weightList)
    avg_acc = calcWeightedAverage(accuracyList, weightList)
    # output_list.append('Times we had no information flow: ' + str(no_information_flow_count) + '\n')
    output_list.append('Average Accuracy: ' + str(avg_acc) + '\n')
    # print('Times we had no information flow: ',str(no_information_flow_count))
    print("Average Accuracy:", avg_acc)
    return avg_acc