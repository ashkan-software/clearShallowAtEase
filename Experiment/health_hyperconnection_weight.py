import gc
import os

import keras.backend as K
import numpy as np
from Experiment.accuracy import accuracy
from Experiment.common import (
    average,
    convert_to_string,
    make_no_information_flow_map,
    make_output_dictionary_hyperconnection_weight,
    make_results_folder,
    save_output,
)
from Experiment.common_MLP_health import (
    batch_size,
    get_model_weights_MLP_health,
    hidden_units,
    init_data,
    num_classes,
    num_iterations,
    num_train_epochs,
    reliability_settings,
)
from Experiment.mlp_DFG_health import (
    default_skip_hyperconnection_config,
    define_DFG_MLP,
)
from Experiment.mlp_ResiliNet_health import MUX_ADDS, define_ResiliNet_MLP
from keras.callbacks import ModelCheckpoint


def define_and_train(
    iteration,
    model_name,
    load_for_inference,
    weight_scheme,
    reliability_setting,
    data,
    num_train_epochs,
    batch_size,
    num_vars,
    num_classes,
    hidden_units,
    verbose,
):
    if model_name == "DFG Hyperconnection Weight":
        model = define_DFG_MLP(
            num_vars,
            num_classes,
            hidden_units,
            reliability_setting=reliability_setting,
            hyperconnection_weights_scheme=weight_scheme,
        )
        model_file = (
            "models/"
            + str(iteration)
            + "_"
            + str(reliability_setting)
            + "_"
            + str(weight_scheme)
            + "health_hyperconnection_DFG.h5"
        )
    else:  # model_name is "ResiliNet Hyperconnection Weight"
        mux_adds_str = "mux_adds" if MUX_ADDS else ""
        model = define_ResiliNet_MLP(
            num_vars,
            num_classes,
            hidden_units,
            reliability_setting=reliability_setting,
            hyperconnection_weights_scheme=weight_scheme,
        )
        model_file = (
            "models/"
            + str(iteration)
            + "_"
            + mux_adds_str
            + str(reliability_setting)
            + "_"
            + str(weight_scheme)
            + "health_hyperconnection_ResiliNet.h5"
        )
    get_model_weights_MLP_health(
        model,
        model_name,
        load_for_inference,
        model_file,
        data,
        num_train_epochs,
        batch_size,
        verbose,
    )
    return model


# runs all 3 failure configurations for all 3 models
if __name__ == "__main__":
    accuracy = accuracy("Health")
    calc_expected_accuracy = accuracy.calc_expected_accuracy
    data, num_vars = init_data()

    no_information_flow_map = make_no_information_flow_map(
        "Health", default_skip_hyperconnection_config
    )
    load_for_inference = False
    # file name with the experiments accuracy output
    mux_adds_str = "mux_adds" if MUX_ADDS else ""
    output_name = "results/health_hyperconnection_weight" + mux_adds_str + ".txt"
    verbose = 2
    model_name = "ResiliNet Hyperconnection Weight"
    hyperconnection_weightedbyreliability_config = 2
    # keep track of output so that output is in order
    output_list = []

    output, weight_schemes = make_output_dictionary_hyperconnection_weight(
        model_name, reliability_settings, num_iterations
    )
    considered_weight_schemes = [1, 2, 3, 4]
    default_reliability_setting = [1, 1, 1]
    make_results_folder()
    for iteration in range(1, num_iterations + 1):
        output_list.append("ITERATION " + str(iteration) + "\n")
        print("ITERATION ", iteration)
        # loop through all the weight schemes
        for weight_scheme in considered_weight_schemes:
            if (
                weight_scheme == 2 or weight_scheme == 3
            ):  # if the weight scheme depends on reliability
                for reliability_setting in reliability_settings:
                    hyperconnection_weight = define_and_train(
                        iteration,
                        model_name,
                        load_for_inference,
                        weight_scheme,
                        reliability_setting,
                        data,
                        num_train_epochs,
                        batch_size,
                        num_vars,
                        num_classes,
                        hidden_units,
                        verbose,
                    )
                    output[model_name][weight_scheme][str(reliability_setting)][
                        iteration - 1
                    ] = calc_expected_accuracy(
                        hyperconnection_weight,
                        no_information_flow_map,
                        reliability_setting,
                        output_list,
                        data=data,
                    )
                    # clear session so that model will recycled back into memory
                    K.clear_session()
                    gc.collect()
                    del hyperconnection_weight
            else:
                hyperconnection_weight = define_and_train(
                    iteration,
                    model_name,
                    load_for_inference,
                    weight_scheme,
                    default_reliability_setting,
                    data,
                    num_train_epochs,
                    batch_size,
                    num_vars,
                    num_classes,
                    hidden_units,
                    verbose,
                )
                for reliability_setting in reliability_settings:
                    output[model_name][weight_scheme][str(reliability_setting)][
                        iteration - 1
                    ] = calc_expected_accuracy(
                        hyperconnection_weight,
                        no_information_flow_map,
                        reliability_setting,
                        output_list,
                        data=data,
                    )
                # clear session so that model will recycled back into memory
                K.clear_session()
                gc.collect()
                del hyperconnection_weight
    # calculate average accuracies
    for reliability_setting in reliability_settings:
        for weight_scheme in weight_schemes:
            hyperconnection_weight_acc = average(
                output[model_name][weight_scheme][str(reliability_setting)]
            )
            output_list.append(
                str(reliability_setting)
                + str(weight_scheme)
                + " "
                + model_name
                + ": "
                + str(hyperconnection_weight_acc)
                + "\n"
            )
            print(
                str(reliability_setting),
                weight_scheme,
                model_name,
                ":",
                hyperconnection_weight_acc,
            )

            hyperconnection_weight_std = np.std(
                output[model_name][weight_scheme][str(reliability_setting)], ddof=1
            )
            output_list.append(
                str(reliability_setting)
                + str(weight_scheme)
                + " "
                + model_name
                + " std: "
                + str(hyperconnection_weight_std)
                + "\n"
            )
            print(
                str(reliability_setting),
                weight_scheme,
                model_name,
                "std:",
                hyperconnection_weight_std,
            )
    save_output(output_name, output_list)
    print(output)
