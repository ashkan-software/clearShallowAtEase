
from Experiment.common_CNN import define_model
from Experiment.accuracy import accuracy
from Experiment.common_CNN_imagenet import init_data, get_model_weights_CNN_imagenet, num_iterations,num_train_examples,num_test_examples, reliability_settings, input_shape, num_classes, alpha, epochs, num_gpus, strides, num_workers
from Experiment.common import average, convert_to_string, make_output_dictionary_average_accuracy, make_results_folder,save_output
import keras.backend as K
import datetime
import gc
import os
import numpy as np
from Experiment.common import make_no_information_flow_map
from Experiment.cnn_DFG_MobileNet import default_skip_hyperconnection_config

import tensorflow as tf
def define_and_train(iteration, model_name, load_for_inference, continue_training, train_generator, val_generator, input_shape, classes, alpha,num_train_examples, epochs,num_gpus, strides, num_workers):
    model, parallel_model, model_file = define_model(iteration, model_name, "imagenet", input_shape, classes, alpha, strides, num_gpus, weights='imagenet')
    model = get_model_weights_CNN_imagenet(model, parallel_model, model_name, load_for_inference, continue_training, model_file, train_generator, val_generator,num_train_examples,epochs, num_gpus, num_workers)
    return model

def calc_accuracy(iteration, model_name, model, no_information_flow_map, reliability_setting, output_list,test_generator, num_test_examples):
    output_list.append(model_name + "\n")
    print(model_name)
    output[model_name][str(reliability_setting)][iteration-1] = calc_expected_accuracy(model,no_information_flow_map,reliability_setting,output_list,test_generator= test_generator,num_test_examples = num_test_examples)


# runs all 3 failure configurations for all 3 models
if __name__ == "__main__":
    accuracy = accuracy("Imagenet")
    calc_expected_accuracy = accuracy.calc_expected_accuracy
    
    train_generator, test_generator = init_data(num_gpus) 
    
    ResiliNet_no_information_flow_map = make_no_information_flow_map("CIFAR/Imagenet", default_skip_hyperconnection_config)
    DFG_no_information_flow_map = make_no_information_flow_map("CIFAR/Imagenet", default_skip_hyperconnection_config)
    Vanilla_no_information_flow_map = make_no_information_flow_map("CIFAR/Imagenet")
    
    load_for_inference = False
    continue_training = False # loads a pre-trained model and improves it with more training
    make_results_folder()
    output_name = 'results' + '/imagenet_average_accuracy_results.txt'
    output_list = []
    
    output = make_output_dictionary_average_accuracy(reliability_settings, num_iterations)

    val_generator = None
    
    for iteration in range(1,num_iterations+1):   
        output_list.append('ITERATION ' + str(iteration) +  '\n')
        print("ITERATION ", iteration)
        ResiliNet = define_and_train(
            iteration = iteration, 
            model_name = "ResiliNet", 
            load_for_inference = load_for_inference, 
            continue_training = continue_training,
            train_generator = train_generator, 
            val_generator = val_generator, 
            input_shape = input_shape, 
            classes = num_classes, 
            alpha = alpha, 
            num_train_examples = num_train_examples,
            epochs = epochs,
            num_gpus = num_gpus,
            strides = strides,
            num_workers = num_workers
            )
        DFG = define_and_train(
            iteration = iteration, 
            model_name = "DFG", 
            load_for_inference = load_for_inference, 
            continue_training = continue_training,
            train_generator = train_generator, 
            val_generator = val_generator, 
            input_shape = input_shape, 
            classes = num_classes, 
            alpha = alpha, 
            num_train_examples = num_train_examples,
            epochs = epochs,
            num_gpus = num_gpus,
            strides = strides,
            num_workers = num_workers
            )
        Vanilla = define_and_train(
            iteration = iteration, 
            model_name = "Vanilla", 
            load_for_inference = load_for_inference, 
            continue_training = continue_training,
            train_generator = train_generator, 
            val_generator = val_generator, 
            input_shape = input_shape, 
            classes = num_classes, 
            alpha = alpha, 
            num_train_examples = num_train_examples,
            epochs = epochs,
            num_gpus = num_gpus,
            strides = strides,
            num_workers = num_workers
            )
        # test models
        for reliability_setting in reliability_settings:
            calc_accuracy(iteration, "ResiliNet", ResiliNet, ResiliNet_no_information_flow_map, reliability_setting, output_list,test_generator, num_test_examples)
            calc_accuracy(iteration, "DFG", DFG, DFG_no_information_flow_map, reliability_setting, output_list,test_generator, num_test_examples)
            calc_accuracy(iteration, "Vanilla", Vanilla, Vanilla_no_information_flow_map, reliability_setting, output_list,test_generator, num_test_examples)
            
        # clear session so that model will recycled back into memory
        K.clear_session()
        gc.collect()
        del DFG
        del ResiliNet
        del Vanilla
   # calculate average accuracies from all expected accuracies
    for reliability_setting in reliability_settings:
        ResiliNet_acc = average(output["ResiliNet"][str(reliability_setting)])
        DFG_acc = average(output["DFG"][str(reliability_setting)])
        Vanilla_acc = average(output["Vanilla"][str(reliability_setting)])

        ResiliNet_std = np.std(output["ResiliNet"][str(reliability_setting)],ddof=1)
        DFG_std = np.std(output["DFG"][str(reliability_setting)],ddof = 1)
        Vanilla_std = np.std(output["Vanilla"][str(reliability_setting)],ddof = 1)

        output_list.append(str(reliability_setting) + " ResiliNet accuracy: " + str(ResiliNet_acc) + '\n')
        output_list.append(str(reliability_setting) + " DFG accuracy: " + str(DFG_acc) + '\n')
        output_list.append(str(reliability_setting) + " Vanilla accuracy: " + str(Vanilla_acc) + '\n')

        output_list.append(str(reliability_setting) + " ResiliNet std: " + str(ResiliNet_std) + '\n')
        output_list.append(str(reliability_setting) + " DFG std: " + str(DFG_std) + '\n')
        output_list.append(str(reliability_setting) + " Vanilla std: " + str(Vanilla_std) + '\n')

        print(str(reliability_setting),"ResiliNet accuracy:",ResiliNet_acc)
        print(str(reliability_setting),"DFG accuracy:",DFG_acc)
        print(str(reliability_setting),"Vanilla accuracy:",Vanilla_acc)

        print(str(reliability_setting),"ResiliNet std:",ResiliNet_std)
        print(str(reliability_setting),"DFG std:",DFG_std)
        print(str(reliability_setting),"Vanilla std:",Vanilla_std)
    save_output(output_name, output_list)
