import gc
import math
import os

import keras.backend as K
import numpy as np
from Experiment.accuracy import accuracy
from Experiment.cnn_DFG_MobileNet import (
    default_skip_hyperconnection_config,
    define_DFG_CNN_MobileNet,
)
from Experiment.cnn_ResiliNet_MobileNet import ResiliNetPlus, define_ResiliNet_CNN_MobileNet
from Experiment.common import (
    average,
    convert_to_string,
    make_no_information_flow_map,
    make_results_folder,
    save_output,
)
from Experiment.common_CNN_cifar import (
    alpha,
    batch_size,
    checkpoint_verbose,
    classes,
    epochs,
    get_model_weights_CNN_cifar,
    init_data,
    input_shape,
    num_gpus,
    num_iterations,
    progress_verbose,
    reliability_settings,
    strides,
    train_datagen,
)
from keras.applications.mobilenet import MobileNet
from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator


def make_output_dictionary(
    model_name,
    reliability_settings,
    num_iterations,
    skip_hyperconnection_configurations,
):
    no_failure, normal, poor, hazardous = convert_to_string(reliability_settings)

    # convert hyperconnection configuration into strings to be used as keys for dictionary
    config = [0] * 5
    for i in range(0, 4):
        config[i] = str(skip_hyperconnection_configurations[i])

    # dictionary to store all the results
    output = {
        model_name: {
            hazardous: {
                config[0]: [0] * num_iterations,
                config[1]: [0] * num_iterations,
                config[2]: [0] * num_iterations,
                config[3]: [0] * num_iterations,
            },
            poor: {
                config[0]: [0] * num_iterations,
                config[1]: [0] * num_iterations,
                config[2]: [0] * num_iterations,
                config[3]: [0] * num_iterations,
            },
            normal: {
                config[0]: [0] * num_iterations,
                config[1]: [0] * num_iterations,
                config[2]: [0] * num_iterations,
                config[3]: [0] * num_iterations,
            },
            no_failure: {
                config[0]: [0] * num_iterations,
                config[1]: [0] * num_iterations,
                config[2]: [0] * num_iterations,
                config[3]: [0] * num_iterations,
            },
        }
    }
    return output


def define_and_train(
    iteration,
    model_name,
    load_for_inference,
    reliability_setting,
    skip_hyperconnection_configuration,
    data,
    batch_size,
    classes,
    input_shape,
    alpha,
    strides,
    train_datagen,
    epochs,
    progress_verbose,
    checkpoint_verbose,
    train_steps_per_epoch,
    val_steps_per_epoch,
    num_gpus,
):
    if model_name == "DFG Hyperconnection Weight Sensitivity":
        model_file = (
            "models/"
            + str(iteration)
            + " "
            + str(skip_hyperconnection_configuration)
            + " "
            + "cifar_skiphyperconnection_sensitivity_DFG.h5"
        )
        model, parallel_model = define_DFG_CNN_MobileNet(
            classes=classes,
            input_shape=input_shape,
            alpha=alpha,
            reliability_setting=reliability_setting,
            skip_hyperconnection_config=skip_hyperconnection_configuration,
            strides=strides,
            num_gpus=num_gpus,
        )
    else:  # model_name is "ResiliNet Hyperconnection Weight Sensitivity"
        mux_adds_str = "mux_adds" if ResiliNetPlus else ""
        model_file = (
            "models/"
            + str(iteration)
            + " "
            + mux_adds_str
            + str(skip_hyperconnection_configuration)
            + " "
            + "cifar_skiphyperconnection_sensitivity_ResiliNet.h5"
        )
        model, parallel_model = define_ResiliNet_CNN_MobileNet(
            classes=classes,
            input_shape=input_shape,
            alpha=alpha,
            reliability_setting=reliability_setting,
            skip_hyperconnection_config=skip_hyperconnection_configuration,
            strides=strides,
            num_gpus=num_gpus,
        )
    get_model_weights_CNN_cifar(
        model,
        parallel_model,
        model_name,
        load_for_inference,
        model_file,
        data,
        train_datagen,
        batch_size,
        epochs,
        progress_verbose,
        checkpoint_verbose,
        train_steps_per_epoch,
        val_steps_per_epoch,
        num_gpus,
    )
    return model


if __name__ == "__main__":
    accuracy = accuracy("CIFAR")
    calc_expected_accuracy = accuracy.calc_expected_accuracy
    data = init_data()

    skip_hyperconnection_configurations = [
        # [e1,IoT]
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
    ]
    model_name = "ResiliNet Hyperconnection Weight Sensitivity"
    default_reliability_setting = [1.0, 1.0, 1.0]
    output = make_output_dictionary(
        model_name,
        reliability_settings,
        num_iterations,
        skip_hyperconnection_configurations,
    )

    no_information_flow_map = {}
    for skip_hyperconnection_configuration in skip_hyperconnection_configurations:
        no_information_flow_map[
            tuple(skip_hyperconnection_configuration)
        ] = make_no_information_flow_map(
            "CIFAR/Imagenet", skip_hyperconnection_configuration
        )

    load_for_inference = False
    train_steps_per_epoch = math.ceil(len(data.train) / batch_size)
    val_steps_per_epoch = math.ceil(len(data.val) / batch_size)

    make_results_folder()
    mux_adds_str = "mux_adds" if ResiliNetPlus else ""
    output_name = (
        "results/cifar_skiphyperconnection_sensitivity_results" + mux_adds_str + ".txt"
    )
    output_list = []
    for iteration in range(1, num_iterations + 1):
        print("iteration:", iteration)
        for skip_hyperconnection_configuration in skip_hyperconnection_configurations:

            model = define_and_train(
                iteration,
                model_name,
                load_for_inference,
                default_reliability_setting,
                skip_hyperconnection_configuration,
                data,
                batch_size,
                classes,
                input_shape,
                alpha,
                strides,
                train_datagen,
                epochs,
                progress_verbose,
                checkpoint_verbose,
                train_steps_per_epoch,
                val_steps_per_epoch,
                num_gpus,
            )
            for reliability_setting in reliability_settings:
                output_list.append(str(reliability_setting) + "\n")
                print(reliability_setting)
                output[model_name][str(reliability_setting)][
                    str(skip_hyperconnection_configuration)
                ][iteration - 1] = calc_expected_accuracy(
                    model,
                    no_information_flow_map[tuple(skip_hyperconnection_configuration)],
                    reliability_setting,
                    output_list,
                    data=data,
                )
            # clear session so that model will recycled back into memory
            K.clear_session()
            gc.collect()
            del model

    for reliability_setting in reliability_settings:
        for skip_hyperconnection_configuration in skip_hyperconnection_configurations:
            output_list.append(str(reliability_setting) + "\n")
            acc = average(
                output[model_name][str(reliability_setting)][
                    str(skip_hyperconnection_configuration)
                ]
            )
            std = np.std(
                output[model_name][str(reliability_setting)][
                    str(skip_hyperconnection_configuration)
                ],
                ddof=1,
            )
            output_list.append(
                str(reliability_setting)
                + str(skip_hyperconnection_configuration)
                + str(acc)
                + "\n"
            )
            output_list.append(
                str(reliability_setting)
                + str(skip_hyperconnection_configuration)
                + str(std)
                + "\n"
            )
            print(str(reliability_setting), acc)
            print(str(reliability_setting), std)
    save_output(output_name, output_list)
    print(output)
