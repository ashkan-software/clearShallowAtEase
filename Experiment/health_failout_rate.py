import gc
import os

import keras.backend as K
import numpy as np
from Experiment.accuracy import accuracy
from Experiment.common import (
    average,
    convert_to_string,
    make_no_information_flow_map,
    make_output_dictionary_failout_rate,
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
from Experiment.mlp_DFG_health import default_skip_hyperconnection_config
from Experiment.mlp_ResiliNet_health import ResiliNetPlus, define_ResiliNet_MLP
from keras.callbacks import ModelCheckpoint


def define_and_train(
    iteration,
    model_name,
    load_for_inference,
    failout_survival_setting,
    data,
    num_train_epochs,
    batch_size,
    num_vars,
    num_classes,
    hidden_units,
    verbose,
):
    mux_adds_str = "mux_adds" if ResiliNetPlus else ""
    model = define_ResiliNet_MLP(
        num_vars,
        num_classes,
        hidden_units,
        failout_survival_setting=failout_survival_setting,
    )
    model_file = (
        "models/"
        + str(iteration)
        + " "
        + mux_adds_str
        + str(failout_survival_setting)
        + "health_failout_rate.h5"
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


def multiply_hyperconnection_weights(
    dropout_like_failout, failout_survival_setting, model
):
    if dropout_like_failout == True:
        nodes = ["fog1_output_layer", "fog2_output_layer", "edge_output_layer"]
        for i, node in enumerate(nodes):
            failout_survival_rate = failout_survival_setting[i]
            # node failed
            layer_name = node
            layer = model.get_layer(name=layer_name)
            layer_weights = layer.get_weights()
            # make new weights for the connections
            new_weights = layer_weights[0] * failout_survival_rate

            # make new weights for biases
            new_bias_weights = layer_weights[1] * failout_survival_rate
            layer.set_weights([new_weights, new_bias_weights])


# runs all 3 failure configurations for all 3 models
if __name__ == "__main__":
    accuracy = accuracy("Health")
    calc_expected_accuracy = accuracy.calc_expected_accuracy
    data, num_vars = init_data()

    load_for_inference = False
    failout_survival_settings = [
        [0.95, 0.95, 0.95],
        [0.9, 0.9, 0.9],
        [0.7, 0.7, 0.7],
        [0.5, 0.5, 0.5],
    ]
    no_information_flow_map = make_no_information_flow_map(
        "Health", default_skip_hyperconnection_config
    )
    # file name with the experiments accuracy output
    mux_adds_str = "mux_adds" if ResiliNetPlus else ""
    output_name = "results/health_failout_rate" + mux_adds_str + ".txt"
    verbose = 2
    # keep track of output so that output is in order
    output_list = []

    output = make_output_dictionary_failout_rate(
        failout_survival_settings, reliability_settings, num_iterations
    )
    dropout_like_failout = False
    make_results_folder()
    for iteration in range(1, num_iterations + 1):
        output_list.append("ITERATION " + str(iteration) + "\n")
        print("ITERATION ", iteration)
        output_list.append("ResiliNet" + "\n")
        # variable failout rate
        for reliability_setting in reliability_settings:
            if reliability_setting == [1, 1, 1]:
                output["Variable Failout 1x"][str(reliability_setting)][
                    iteration - 1
                ] = 0
                continue
            ResiliNet_failout_rate_variable = define_and_train(
                iteration,
                "Variable Failout 1x",
                load_for_inference,
                reliability_setting,
                data,
                num_train_epochs,
                batch_size,
                num_vars,
                num_classes,
                hidden_units,
                verbose,
            )
            multiply_hyperconnection_weights(
                dropout_like_failout,
                reliability_setting,
                ResiliNet_failout_rate_variable,
            )
            output["Variable Failout 1x"][str(reliability_setting)][
                iteration - 1
            ] = calc_expected_accuracy(
                ResiliNet_failout_rate_variable,
                no_information_flow_map,
                reliability_setting,
                output_list,
                data=data,
            )

            # clear session so that model will recycled back into memory
            K.clear_session()
            gc.collect()
            del ResiliNet_failout_rate_variable
        # fixed failout rate
        for failout_survival_setting in failout_survival_settings:
            ResiliNet_failout_rate_fixed = define_and_train(
                iteration,
                "Fixed Failout 1x",
                load_for_inference,
                failout_survival_setting,
                data,
                num_train_epochs,
                batch_size,
                num_vars,
                num_classes,
                hidden_units,
                verbose,
            )
            multiply_hyperconnection_weights(
                dropout_like_failout,
                failout_survival_setting,
                ResiliNet_failout_rate_fixed,
            )

            for reliability_setting in reliability_settings:
                output_list.append(str(reliability_setting) + "\n")
                print(reliability_setting)
                output[str(failout_survival_setting)][str(reliability_setting)][
                    iteration - 1
                ] = calc_expected_accuracy(
                    ResiliNet_failout_rate_fixed,
                    no_information_flow_map,
                    reliability_setting,
                    output_list,
                    data=data,
                )
            # clear session so that model will recycled back into memory
            K.clear_session()
            gc.collect()
            del ResiliNet_failout_rate_fixed

    # calculate average accuracies for variable failout rate
    for reliability_setting in reliability_settings:
        ResiliNet_failout_rate_acc = average(
            output["Variable Failout 1x"][str(reliability_setting)]
        )
        output_list.append(
            str(reliability_setting)
            + " Variable Failout 1x: "
            + str(ResiliNet_failout_rate_acc)
            + "\n"
        )
        print(reliability_setting, "Variable Failout 1x:", ResiliNet_failout_rate_acc)

        ResiliNet_failout_rate_std = np.std(
            output["Variable Failout 1x"][str(reliability_setting)], ddof=1
        )
        output_list.append(
            str(reliability_setting)
            + " Variable Failout 1x std: "
            + str(ResiliNet_failout_rate_std)
            + "\n"
        )
        print(
            str(reliability_setting),
            " Variable Failout 1x std:",
            ResiliNet_failout_rate_std,
        )
    # calculate average accuracies for fixed failout rate
    for failout_survival_setting in failout_survival_settings:
        print(failout_survival_setting)
        for reliability_setting in reliability_settings:
            ResiliNet_failout_rate_acc = average(
                output[str(failout_survival_setting)][str(reliability_setting)]
            )
            output_list.append(
                str(failout_survival_setting)
                + str(reliability_setting)
                + " Fixed Failout: "
                + str(ResiliNet_failout_rate_acc)
                + "\n"
            )
            print(
                failout_survival_setting,
                reliability_setting,
                "Fixed Failout:",
                ResiliNet_failout_rate_acc,
            )

            ResiliNet_failout_rate_std = np.std(
                output[str(failout_survival_setting)][str(reliability_setting)], ddof=1
            )
            output_list.append(
                str(reliability_setting)
                + " Fixed Failout std: "
                + str(ResiliNet_failout_rate_std)
                + "\n"
            )
            print(
                str(reliability_setting),
                "Fixed Failout std:",
                ResiliNet_failout_rate_std,
            )

    # write experiments output to file
    save_output(output_name, output_list)
    print(output)
