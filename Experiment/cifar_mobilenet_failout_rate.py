
from keras.datasets import cifar10
from keras.applications.mobilenet import MobileNet
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import math
import os 
from Experiment.cnn_ResiliNet_MobileNet import define_ResiliNet_CNN_MobileNet, MUX_ADDS
from Experiment.accuracy import accuracy
from Experiment.common_CNN_cifar import init_data, get_model_weights_CNN_cifar, num_iterations, classes, reliability_settings, train_datagen, batch_size, epochs, progress_verbose, checkpoint_verbose, alpha, input_shape, strides, num_gpus
from Experiment.common import average, make_results_folder, make_output_dictionary_failout_rate, save_output
import numpy as np
import gc
from Experiment.common import make_no_information_flow_map
from Experiment.cnn_DFG_MobileNet import default_skip_hyperconnection_config

def define_and_train(iteration, model_name, load_for_inference, failout_survival_setting, training_data, training_labels, val_data, val_labels, batch_size, classes, input_shape, alpha, strides, train_datagen, epochs, progress_verbose, checkpoint_verbose, train_steps_per_epoch, val_steps_per_epoch, num_gpus):
    mux_adds_str = "mux_adds" if MUX_ADDS else "" 
    model_file = 'models/' + str(iteration) + " " +mux_adds_str+ str(failout_survival_setting) + 'cifar_failout_rate.h5'
    model, parallel_model = define_ResiliNet_CNN_MobileNet(classes=classes,input_shape = input_shape,alpha = alpha,failout_survival_setting=failout_survival_setting, strides = strides, num_gpus=num_gpus)
    get_model_weights_CNN_cifar(model, parallel_model, model_name, load_for_inference, model_file, training_data, training_labels, val_data, val_labels, train_datagen, batch_size, epochs, progress_verbose, checkpoint_verbose, train_steps_per_epoch, val_steps_per_epoch, num_gpus)
    return model

def multiply_hyperconnection_weights(dropout_like_failout, failout_survival_setting, model):
    if dropout_like_failout == True:
        nodes = ["conv_pw_8","conv_pw_3"]
        for i, node in enumerate(nodes):
            failout_survival_rate = failout_survival_setting[i]
            # node failed
            layer_name = node
            layer = model.get_layer(name=layer_name)
            layer_weights = layer.get_weights()
            # make new weights for the connections
            new_weights = layer_weights[0] * failout_survival_rate
            layer.set_weights([new_weights])

# ResiliNet variable failout experiment
if __name__ == "__main__":
    accuracy = accuracy("CIFAR")
    calc_expected_accuracy = accuracy.calc_expected_accuracy
    training_data, test_data, training_labels, test_labels, val_data, val_labels = init_data() 

    output_list = []

    load_for_inference = False
    train_steps_per_epoch = math.ceil(len(training_data) / batch_size)
    val_steps_per_epoch = math.ceil(len(val_data) / batch_size)
    no_information_flow_map = make_no_information_flow_map("CIFAR/Imagenet", default_skip_hyperconnection_config)
    failout_survival_settings = [
        [.95,.95],
        [.9,.9],
        [.7,.7],
        [.5,.5],
    ]
    dropout_like_failout = False
    output = make_output_dictionary_failout_rate(failout_survival_settings, reliability_settings, num_iterations)
    make_results_folder()
    mux_adds_str = "mux_adds" if MUX_ADDS else "" 
    output_name = 'results/cifar_failout_results'+mux_adds_str+'.txt'
    for iteration in range(1,num_iterations+1):
        output_list.append('ITERATION ' + str(iteration) +  '\n')
        print("iteration:",iteration)
        output_list.append('ResiliNet' + '\n') 
        # variable failout rate  
        for reliability_setting in reliability_settings:
            if reliability_setting == [1,1]:
                output["Variable Failout 1x"][str(reliability_setting)][iteration-1] = 0
                continue
            ResiliNet_failout_rate_variable = define_and_train(iteration, "Variable Failout 1x", load_for_inference, reliability_setting, training_data, training_labels, val_data, val_labels, batch_size, classes, input_shape, alpha, strides, train_datagen, epochs, progress_verbose, checkpoint_verbose, train_steps_per_epoch, val_steps_per_epoch, num_gpus)
            multiply_hyperconnection_weights(dropout_like_failout, reliability_setting, ResiliNet_failout_rate_variable)
            output["Variable Failout 1x"][str(reliability_setting)][iteration-1] = calc_expected_accuracy(ResiliNet_failout_rate_variable, no_information_flow_map, reliability_setting,output_list, training_labels= training_labels, test_data= test_data, test_labels= test_labels)
            
            # clear session so that model will recycled back into memory
            K.clear_session()
            gc.collect()
            del ResiliNet_failout_rate_variable
        # fixed failout rate
        for failout_survival_setting in failout_survival_settings:
            ResiliNet_failout_rate_fixed = define_and_train(iteration, "Fixed Failout 1x", load_for_inference, failout_survival_setting, training_data, training_labels, val_data, val_labels, batch_size, classes, input_shape, alpha, strides, train_datagen, epochs, progress_verbose, checkpoint_verbose, train_steps_per_epoch, val_steps_per_epoch, num_gpus)
            multiply_hyperconnection_weights(dropout_like_failout, failout_survival_setting, ResiliNet_failout_rate_fixed)   
                
            for reliability_setting in reliability_settings:
                output_list.append(str(reliability_setting)+ '\n')
                print(reliability_setting)
                output[str(failout_survival_setting)][str(reliability_setting)][iteration-1] = calc_expected_accuracy(ResiliNet_failout_rate_fixed, no_information_flow_map, reliability_setting,output_list,training_labels= training_labels, test_data= test_data, test_labels= test_labels)
            # clear session so that model will recycled back into memory
            K.clear_session()
            gc.collect()
            del ResiliNet_failout_rate_fixed
    
     # calculate average accuracies for variable failout rate
    for reliability_setting in reliability_settings:
        ResiliNet_failout_rate_acc = average(output["Variable Failout 1x"][str(reliability_setting)])
        output_list.append(str(reliability_setting) + " Variable Failout 1x: " + str(ResiliNet_failout_rate_acc) + '\n')
        print(reliability_setting,"Variable Failout 1x:",ResiliNet_failout_rate_acc)  

        ResiliNet_failout_rate_std = np.std(output["Variable Failout 1x"][str(reliability_setting)],ddof=1)
        output_list.append(str(reliability_setting) + " Variable Failout 1x std: " + str(ResiliNet_failout_rate_std) + '\n')
        print(str(reliability_setting), " Variable Failout 1x std:",ResiliNet_failout_rate_std)
    # calculate average accuracies for fixed failout rate
    for failout_survival_setting in failout_survival_settings:
        print(failout_survival_setting)
        for reliability_setting in reliability_settings:
            ResiliNet_failout_rate_acc = average(output[str(failout_survival_setting)][str(reliability_setting)])
            output_list.append(str(failout_survival_setting) + str(reliability_setting) + " Fixed Failout: " + str(ResiliNet_failout_rate_acc) + '\n')
            print(failout_survival_setting,reliability_setting,"Fixed Failout:",ResiliNet_failout_rate_acc)  

            ResiliNet_failout_rate_std = np.std(output[str(failout_survival_setting)][str(reliability_setting)],ddof=1)
            output_list.append(str(reliability_setting) + " Fixed Failout std: " + str(ResiliNet_failout_rate_std) + '\n')
            print(str(reliability_setting), "Fixed Failout std:",ResiliNet_failout_rate_std)
    save_output(output_name, output_list)
    print(output)
