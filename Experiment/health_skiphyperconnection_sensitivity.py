import gc
import os

import keras.backend as K
import numpy as np
from Experiment.accuracy import accuracy
from Experiment.common import (
    average,
    convert_to_string,
    make_no_information_flow_map,
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


def make_output_dictionary(
    model_name,
    reliability_settings,
    num_iterations,
    skip_hyperconnection_configurations,
):
    no_failure, normal, poor, hazardous = convert_to_string(reliability_settings)

    # convert hyperconnection configuration into strings to be used as keys for dictionary
    config = [0] * 9
    for i in range(0, 8):
        config[i] = str(skip_hyperconnection_configurations[i])

    # dictionary to store all the results
    output = {
        model_name: {
            hazardous: {
                config[0]: [0] * num_iterations,
                config[1]: [0] * num_iterations,
                config[2]: [0] * num_iterations,
                config[3]: [0] * num_iterations,
                config[4]: [0] * num_iterations,
                config[5]: [0] * num_iterations,
                config[6]: [0] * num_iterations,
                config[7]: [0] * num_iterations,
            },
            poor: {
                config[0]: [0] * num_iterations,
                config[1]: [0] * num_iterations,
                config[2]: [0] * num_iterations,
                config[3]: [0] * num_iterations,
                config[4]: [0] * num_iterations,
                config[5]: [0] * num_iterations,
                config[6]: [0] * num_iterations,
                config[7]: [0] * num_iterations,
            },
            normal: {
                config[0]: [0] * num_iterations,
                config[1]: [0] * num_iterations,
                config[2]: [0] * num_iterations,
                config[3]: [0] * num_iterations,
                config[4]: [0] * num_iterations,
                config[5]: [0] * num_iterations,
                config[6]: [0] * num_iterations,
                config[7]: [0] * num_iterations,
            },
            no_failure: {
                config[0]: [0] * num_iterations,
                config[1]: [0] * num_iterations,
                config[2]: [0] * num_iterations,
                config[3]: [0] * num_iterations,
                config[4]: [0] * num_iterations,
                config[5]: [0] * num_iterations,
                config[6]: [0] * num_iterations,
                config[7]: [0] * num_iterations,
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
    num_train_epochs,
    batch_size,
    num_vars,
    num_classes,
    hidden_units,
    verbose,
):
    if model_name == "DFG Hyperconnection Weight Sensitivity":
        model = define_DFG_MLP(
            num_vars,
            num_classes,
            hidden_units,
            reliability_setting=reliability_setting,
            skip_hyperconnection_config=skip_hyperconnection_configuration,
        )
        model_file = (
            "models/"
            + str(iteration)
            + " "
            + str(skip_hyperconnection_configuration)
            + " "
            + "health_skiphyperconnection_sensitivity_DFG.h5"
        )
    else:  # model_name is "ResiliNet Hyperconnection Weight Sensitivity"
        mux_adds_str = "mux_adds" if MUX_ADDS else ""
        model = define_ResiliNet_MLP(
            num_vars,
            num_classes,
            hidden_units,
            reliability_setting=reliability_setting,
            skip_hyperconnection_config=skip_hyperconnection_configuration,
        )
        model_file = (
            "models/"
            + str(iteration)
            + " "
            + mux_adds_str
            + str(skip_hyperconnection_configuration)
            + " "
            + "health_skiphyperconnection_sensitivity_ResiliNet.h5"
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


def calc_accuracy(
    iteration,
    model_name,
    model,
    no_information_flow_map,
    reliability_setting,
    skip_hyperconnection_configuration,
    output_list,
    data,
):
    output_list.append(model_name + "\n")
    print(model_name)
    output[model_name][str(reliability_setting)][
        str(skip_hyperconnection_configuration)
    ][iteration - 1] = calc_expected_accuracy(
        model,
        no_information_flow_map,
        reliability_setting,
        output_list,
        data=data,
    )


if __name__ == "__main__":
    accuracy = accuracy("Health")
    calc_expected_accuracy = accuracy.calc_expected_accuracy
    data, num_vars = init_data()

    skip_hyperconnection_configurations = [
        # [f2,e1,g1]
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]
    default_reliability_setting = [1.0, 1.0, 1.0]
    no_information_flow_map = {}
    for skip_hyperconnection_configuration in skip_hyperconnection_configurations:
        no_information_flow_map[
            tuple(skip_hyperconnection_configuration)
        ] = make_no_information_flow_map("Health", skip_hyperconnection_configuration)

    load_for_inference = False
    mux_adds_str = "mux_adds" if MUX_ADDS else ""
    output_name = (
        "results/health_skiphyperconnection_sensitivity" + mux_adds_str + ".txt"
    )

    verbose = 2
    # keep track of output so that output is in order
    output_list = []
    model_name = "ResiliNet Hyperconnection Weight Sensitivity"
    output = make_output_dictionary(
        model_name,
        reliability_settings,
        num_iterations,
        skip_hyperconnection_configurations,
    )
    make_results_folder()
    for iteration in range(1, num_iterations + 1):
        output_list.append("ITERATION " + str(iteration) + "\n")
        print("ITERATION ", iteration)
        for skip_hyperconnection_configuration in skip_hyperconnection_configurations:

            weight_sesitivity = define_and_train(
                iteration,
                model_name,
                load_for_inference,
                default_reliability_setting,
                skip_hyperconnection_configuration,
                data,
                num_train_epochs,
                batch_size,
                num_vars,
                num_classes,
                hidden_units,
                verbose,
            )
            # test models
            for reliability_setting in reliability_settings:
                print(reliability_setting)
                output_list.append(str(reliability_setting) + "\n")
                calc_accuracy(
                    iteration,
                    model_name,
                    weight_sesitivity,
                    no_information_flow_map[tuple(skip_hyperconnection_configuration)],
                    reliability_setting,
                    skip_hyperconnection_configuration,
                    output_list,
                    data,
                )
            K.clear_session()
            gc.collect()
            del weight_sesitivity

    for reliability_setting in reliability_settings:
        output_list.append(str(reliability_setting) + "\n")
        for skip_hyperconnection_configuration in skip_hyperconnection_configurations:
            output_list.append(str(skip_hyperconnection_configuration) + "\n")
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
            # write to output list
            output_list.append(
                str(reliability_setting)
                + " "
                + str(skip_hyperconnection_configuration)
                + " accuracy: "
                + str(acc)
                + "\n"
            )
            print(
                str(reliability_setting),
                str(skip_hyperconnection_configuration),
                "accuracy:",
                acc,
            )
            output_list.append(
                str(reliability_setting)
                + " "
                + str(skip_hyperconnection_configuration)
                + " std: "
                + str(std)
                + "\n"
            )
            print(
                str(reliability_setting),
                str(skip_hyperconnection_configuration),
                "std:",
                std,
            )

    save_output(output_name, output_list)
    print(output)
