
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import math
import os 
from Experiment.cnn_DFG_ResNet import define_DFG_CNN_ResNet
from Experiment.cnn_ResiliNet_ResNet import define_ResiliNet_CNN_ResNet, MUX_ADDS
from Experiment.accuracy import accuracy
from Experiment.common_CNN_cifar import init_data, get_model_weights_CNN_cifar, num_iterations, classes, reliability_settings, train_datagen, batch_size, epochs, progress_verbose, checkpoint_verbose, alpha, input_shape, strides, num_gpus
from Experiment.common import average, make_results_folder, convert_to_string, write_n_upload, make_results_folder
import numpy as np
import gc
from Experiment.common import make_no_information_flow_map
from Experiment.cnn_DFG_ResNet import default_skip_hyperconnection_config

def make_output_dictionary(model_name, reliability_settings, num_iterations, skip_hyperconnection_configurations):
    no_failure, normal, poor, hazardous = convert_to_string(reliability_settings)

    # convert hyperconnection configuration into strings to be used as keys for dictionary
    config = [0] * 5
    for i in range(0,4):
        config[i] = str(skip_hyperconnection_configurations[i])

    # dictionary to store all the results
    output = {
        model_name:
        {
            hazardous:
            {
                config[0]:[0] * num_iterations,
                config[1]:[0] * num_iterations,
                config[2]:[0] * num_iterations,
                config[3]:[0] * num_iterations
            },
            poor:
            {
                config[0]:[0] * num_iterations,
                config[1]:[0] * num_iterations,
                config[2]:[0] * num_iterations,
                config[3]:[0] * num_iterations
            },
            normal:
            {
                config[0]:[0] * num_iterations,
                config[1]:[0] * num_iterations,
                config[2]:[0] * num_iterations,
                config[3]:[0] * num_iterations
            },
            no_failure:
            {
                config[0]:[0] * num_iterations,
                config[1]:[0] * num_iterations,
                config[2]:[0] * num_iterations,
                config[3]:[0] * num_iterations
            },
        }
    }
    return output

def define_and_train(iteration, model_name, load_for_inference, reliability_setting, skip_hyperconnection_configuration, training_data, training_labels, val_data, val_labels, batch_size, classes, input_shape, alpha, strides, train_datagen, epochs, progress_verbose, checkpoint_verbose, train_steps_per_epoch, val_steps_per_epoch, num_gpus):
    if model_name == "DFG Hyperconnection Weight Sensitivity":
        model_file = 'models/' + str(iteration) + " " + str(skip_hyperconnection_configuration) + " " + 'cifar_resnet_skiphyperconnection_sensitivity_DFG.h5'
        model, parallel_model = define_DFG_CNN_ResNet(input_shape=input_shape, classes=classes, block='basic', residual_unit='v2',
                                    repetitions=[2, 2, 2, 2], initial_filters=64, activation='softmax', include_top=True,
                                    input_tensor=None, dropout=None, transition_dilation_rate=(1, 1),
                                    initial_strides=(2, 2), initial_kernel_size=(7, 7), initial_pooling='max',
                                    final_pooling=None, top='classification',
                                    skip_hyperconnection_config = skip_hyperconnection_configuration, # binary representating if a skip hyperconnection is alive [e1,IoT]
                                    reliability_setting=reliability_setting, # reliability of a node between 0 and 1 [f1,e1]
                                    hyperconnection_weights_scheme = 1,
                                    num_gpus = num_gpus)
    else: # model_name is "ResiliNet Hyperconnection Weight Sensitivity"
        mux_adds_str = "mux_adds" if MUX_ADDS else "" 
        model_file = 'models/' + str(iteration) + " " +mux_adds_str+ str(skip_hyperconnection_configuration) + " " + 'cifar_resnet_skiphyperconnection_sensitivity_ResiliNet.h5'
        model, parallel_model = define_ResiliNet_CNN_ResNet(input_shape=input_shape, classes=classes, block='basic', residual_unit='v2',
                                repetitions=[2, 2, 2, 2], initial_filters=64, activation='softmax', include_top=True,
                                input_tensor=None, dropout=None, transition_dilation_rate=(1, 1),
                                initial_strides=(2, 2), initial_kernel_size=(7, 7), initial_pooling='max',
                                final_pooling=None, top='classification',
                                failout_survival_setting = [.95,.95],
                                skip_hyperconnection_config = skip_hyperconnection_configuration, 
                                reliability_setting=reliability_setting, 
                                hyperconnection_weights_scheme = 1,
                                num_gpus = num_gpus)
    get_model_weights_CNN_cifar(model, parallel_model, model_name, load_for_inference, model_file, training_data, training_labels, val_data, val_labels, train_datagen, batch_size, epochs, progress_verbose, checkpoint_verbose, train_steps_per_epoch, val_steps_per_epoch, num_gpus)
    return model

if __name__ == "__main__":
    accuracy = accuracy("ResNet")
    calc_expected_accuracy = accuracy.calc_expected_accuracy
    training_data, test_data, training_labels, test_labels, val_data, val_labels = init_data() 
    
    skip_hyperconnection_configurations = [
        # [e1,IoT]
        [0,0],
        [1,0],
        [0,1],
        [1,1],
    ]
    model_name = "ResiliNet Hyperconnection Weight Sensitivity"
    default_reliability_setting = [1.0,1.0,1.0]
    output = make_output_dictionary(model_name, reliability_settings, num_iterations, skip_hyperconnection_configurations)
    
    no_information_flow_map = {}
    for skip_hyperconnection_configuration in skip_hyperconnection_configurations:
        no_information_flow_map[tuple(skip_hyperconnection_configuration)] = make_no_information_flow_map("ResNet", skip_hyperconnection_configuration)
    
    load_for_inference = False
    train_steps_per_epoch = math.ceil(len(training_data) / batch_size)
    val_steps_per_epoch = math.ceil(len(val_data) / batch_size)
    
    make_results_folder()
    mux_adds_str = "mux_adds" if MUX_ADDS else "" 
    output_name = 'results/cifar_resnet_skiphyperconnection_sensitivity_results'+mux_adds_str+'.txt'
    output_list = []
    for iteration in range(1,num_iterations+1):
        print("iteration:",iteration)
        for skip_hyperconnection_configuration in skip_hyperconnection_configurations:
            
            model = define_and_train(iteration, model_name, load_for_inference, default_reliability_setting, skip_hyperconnection_configuration, training_data, training_labels, val_data, val_labels, batch_size, classes, input_shape, alpha, strides, train_datagen, epochs, progress_verbose, checkpoint_verbose, train_steps_per_epoch, val_steps_per_epoch, num_gpus)
            for reliability_setting in reliability_settings:
                output_list.append(str(reliability_setting) + '\n')
                print(reliability_setting)
                output[model_name][str(reliability_setting)][str(skip_hyperconnection_configuration)][iteration-1] = calc_expected_accuracy(model, no_information_flow_map[tuple(skip_hyperconnection_configuration)],reliability_setting,output_list, training_labels= training_labels, test_data= test_data, test_labels= test_labels)
            # clear session so that model will recycled back into memory
            K.clear_session()
            gc.collect()
            del model
    
    for reliability_setting in reliability_settings:
        for skip_hyperconnection_configuration in skip_hyperconnection_configurations:
            output_list.append(str(reliability_setting) + '\n')
            acc = average(output[model_name][str(reliability_setting)][str(skip_hyperconnection_configuration)])
            std = np.std(output[model_name][str(reliability_setting)][str(skip_hyperconnection_configuration)],ddof=1)
            output_list.append(str(reliability_setting) + str(skip_hyperconnection_configuration) + str(acc) + '\n')
            output_list.append(str(reliability_setting) + str(skip_hyperconnection_configuration) + str(std) + '\n')
            print(str(reliability_setting),acc)
            print(str(reliability_setting), std)
    write_n_upload(output_name, output_list)
    print(output)