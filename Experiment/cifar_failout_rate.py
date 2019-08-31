
from keras.datasets import cifar10
from keras.applications.mobilenet import MobileNet
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import math
import os 
from Experiment.cnn_deepFogGuardPlus import define_deepFogGuardPlus_CNN
from Experiment.FailureIteration import calculateExpectedAccuracy
from Experiment.common_exp_methods_CNN_cifar import init_data, init_common_experiment_params
from Experiment.common_exp_methods import make_results_folder, make_output_dictionary_failout_rate, write_n_upload
import numpy as np
from Experiment.utility import average, get_model_weights_CNN
import gc


def define_and_train(iteration, model_name, load_model, survivability_setting, training_data, training_labels, val_data, val_labels, batch_size, classes, input_shape, alpha, train_datagen, epochs, progress_verbose, checkpoint_verbose, train_steps_per_epoch, val_steps_per_epoch):
    model_file = str(iteration) + " " + str(failout_survival_setting) + 'cifar_failout_rate.h5'
    model = define_deepFogGuardPlus_CNN(classes=classes,input_shape = input_shape,alpha = alpha,failout_survival_setting=survivability_setting)
    get_model_weights_CNN(model, model_name, load_model, model_file, training_data, training_labels, val_data, val_labels, train_datagen, batch_size, epochs, progress_verbose, checkpoint_verbose, train_steps_per_epoch, val_steps_per_epoch)
    return model

def multiply_hyperconnection_weights(dropout_like_failout, failout_survival_setting, model):
    if dropout_like_failout == True:
        nodes = ["conv_pw_3","conv_pw_8"]
        for i, node in enumerate(nodes):
            survival_rate = failout_survival_setting[i]
            # node failed
            layer_name = node
            layer = model.get_layer(name=layer_name)
            layer_weights = layer.get_weights()
            # make new weights for the connections
            new_weights = layer_weights[0] * survival_rate
            layer.set_weights([new_weights])

# deepFogGuard Plus variable dropout experiment
if __name__ == "__main__":
    
    training_data, test_data, training_labels, test_labels, val_data, val_labels = init_data() 

    num_iterations, classes, survivability_settings, train_datagen, batch_size, epochs, progress_verbose, checkpoint_verbose, use_GCP, alpha, input_shape = init_common_experiment_params() 
    output_list = []

    load_model = False
    train_steps_per_epoch = math.ceil(len(training_data) / batch_size)
    val_steps_per_epoch = math.ceil(len(val_data) / batch_size)
    
    failout_survival_settings = [
        [.95,.95],
        [.9,.9],
        [.7,.7],
        [.5,.5],
    ]
    dropout_like_failout = False
    output = make_output_dictionary_failout_rate(failout_survival_settings, survivability_settings, num_iterations)
    make_results_folder()
    output_name = 'results/cifar_nodewise_dropout_results.txt'
    for iteration in range(1,num_iterations+1):
        output_list.append('ITERATION ' + str(iteration) +  '\n')
        print("iteration:",iteration)
        output_list.append('ResiliNet' + '\n') 
        # variable failout rate  
        for survivability_setting in survivability_settings:
            # # node-wise dropout
            ResiliNet_failout_rate_variable = define_and_train(iteration, "Variable Failout 1x", load_model, survivability_setting, training_data, training_labels, val_data, val_labels, batch_size, classes, input_shape, alpha, train_datagen, epochs, progress_verbose, checkpoint_verbose, train_steps_per_epoch, val_steps_per_epoch)
            multiply_hyperconnection_weights(dropout_like_failout, survivability_setting, ResiliNet_failout_rate_variable)
            output_list.append(str(survivability_setting) + '\n')
            output["Variable Failout 1x"][str(survivability_setting)][iteration-1] = calculateExpectedAccuracy(ResiliNet_failout_rate_variable, survivability_setting,output_list, training_labels, test_data, test_labels)
            
            # clear session so that model will recycled back into memory
            K.clear_session()
            gc.collect()
            del ResiliNet_failout_rate_variable
        # fixed failout rate
        for failout_survival_setting in failout_survival_settings:
            # node-wise dropout
            ResiliNet_failout_rate_fixed = define_and_train(iteration, "Fixed Failout 1x", load_model, survivability_setting, training_data, training_labels, val_data, val_labels, batch_size, classes, input_shape, alpha, train_datagen, epochs, progress_verbose, checkpoint_verbose, train_steps_per_epoch, val_steps_per_epoch)
            multiply_hyperconnection_weights(dropout_like_failout, failout_survival_setting, ResiliNet_failout_rate_fixed)   
                
            for survivability_setting in survivability_settings:
                output_list.append(str(survivability_setting)+ '\n')
                print(survivability_setting)
                output[str(failout_survival_setting)][str(survivability_setting)][iteration-1] = calculateExpectedAccuracy(ResiliNet_failout_rate_fixed,survivability_setting,output_list,training_labels,test_data,test_labels)
                K.set_learning_phase(0)
            # clear session so that model will recycled back into memory
            K.clear_session()
            gc.collect()
            del ResiliNet_failout_rate_fixed
    
     # calculate average accuracies for variable failout rate
    for survivability_setting in survivability_settings:
        ResiliNet_failout_rate_acc = average(output["Variable Failout 1x"][str(survivability_setting)])
        output_list.append(str(survivability_setting) + " Variable Failout 1x: " + str(ResiliNet_failout_rate_acc) + '\n')
        print(survivability_setting,"Variable Failout 1x:",ResiliNet_failout_rate_acc)  

        ResiliNet_failout_rate_std = np.std(output["Variable Failout 1x"][str(survivability_setting)],ddof=1)
        output_list.append(str(survivability_setting) + " Variable Failout 1x std: " + str(ResiliNet_failout_rate_std) + '\n')
        print(str(survivability_setting), " Variable Failout 1x std:",ResiliNet_failout_rate_std)
    # calculate average accuracies for fixed failout rate
    for failout_survival_setting in failout_survival_settings:
        print(failout_survival_setting)
        for survivability_setting in survivability_settings:
            ResiliNet_failout_rate_acc = average(output[str(failout_survival_setting)][str(survivability_setting)])
            output_list.append(str(failout_survival_setting) + str(survivability_setting) + " Fixed Failout: " + str(ResiliNet_failout_rate_acc) + '\n')
            print(failout_survival_setting,survivability_setting,"Fixed Failout:",ResiliNet_failout_rate_acc)  

            ResiliNet_failout_rate_std = np.std(output[str(failout_survival_setting)][str(survivability_setting)],ddof=1)
            output_list.append(str(survivability_setting) + " Fixed Failout std: " + str(ResiliNet_failout_rate_std) + '\n')
            print(str(survivability_setting), "Fixed Failout std:",ResiliNet_failout_rate_std)
    write_n_upload(output_name, output_list, use_GCP)
    print(output)