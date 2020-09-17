import datetime
import gc
import os

import keras.backend as K
import numpy as np
from Experiment.accuracy import accuracy
from Experiment.common import (
    average,
    convert_to_string,
    make_no_information_flow_map,
    make_output_dictionary_average_accuracy,
    save_output,
)
from Experiment.common_MLP_camera import (
    batch_size,
    epochs,
    get_model_weights_MLP_camera,
    hidden_units,
    init_data,
    input_shape,
    num_classes,
    num_iterations,
    reliability_settings,
)
from Experiment.mlp_DFG_camera import (
    default_skip_hyperconnection_config,
    define_DFG_MLP,
)
from Experiment.mlp_ResiliNet_camera import ResiliNetPlus, define_ResiliNet_MLP
from Experiment.mlp_Vanilla_camera import define_vanilla_model_MLP


def define_and_train(
    iteration,
    model_name,
    load_for_inference,
    data,
    input_shape,
    num_classes,
    hidden_units,
    verbose,
    batch_size,
    epochs,
):
    # ResiliNet
    if model_name == "ResiliNet":
        mux_adds_str = "mux_adds" if ResiliNetPlus else ""
        model = define_ResiliNet_MLP(input_shape, num_classes, hidden_units)
        model_file = (
            "models/"
            + "Camera"
            + str(iteration)
            + mux_adds_str
            + "average_accuracy_ResiliNet.h5"
        )
    # DFG
    if model_name == "DFG":
        model = define_DFG_MLP(input_shape, num_classes, hidden_units)
        model_file = "models/" + "Camera" + str(iteration) + "average_accuracy_DFG.h5"
    # Vanilla model
    if model_name == "Vanilla":
        model = define_vanilla_model_MLP(input_shape, num_classes, hidden_units)
        model_file = (
            "models/" + "Camera" + str(iteration) + "average_accuracy_vanilla.h5"
        )

    get_model_weights_MLP_camera(
        model,
        model_name,
        load_for_inference,
        model_file,
        data,
        epochs,
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
    output_list,
    data,
):
    output_list.append(model_name + "\n")
    print(model_name)
    output[model_name][str(reliability_setting)][
        iteration - 1
    ] = calc_expected_accuracy(
        model,
        no_information_flow_map,
        reliability_setting,
        output_list,
        data=data,
    )


# runs all 3 failure configurations for all 3 models
if __name__ == "__main__":
    accuracy = accuracy("Camera")
    calc_expected_accuracy = accuracy.calc_expected_accuracy

    data = init_data()

    ResiliNet_no_information_flow_map = make_no_information_flow_map(
        "Camera", default_skip_hyperconnection_config
    )
    DFG_no_information_flow_map = make_no_information_flow_map(
        "Camera", default_skip_hyperconnection_config
    )
    Vanilla_no_information_flow_map = make_no_information_flow_map("Camera")

    load_for_inference = False

    # file name with the experiments accuracy output
    mux_adds_str = "mux_adds" if ResiliNetPlus else ""
    output_name = "results/camera_average_accuracy" + mux_adds_str + ".txt"
    verbose = 2

    # keep track of output so that output is in order
    output_list = []

    output = make_output_dictionary_average_accuracy(
        reliability_settings, num_iterations
    )

    # make folder for outputs
    if not os.path.exists("results/"):
        os.mkdir("results/")
    for iteration in range(1, num_iterations + 1):
        output_list.append("ITERATION " + str(iteration) + "\n")
        print("ITERATION ", iteration)
        ResiliNet = define_and_train(
            iteration,
            "ResiliNet",
            load_for_inference,
            data,
            input_shape,
            num_classes,
            hidden_units,
            verbose,
            batch_size,
            epochs,
        )
        DFG = define_and_train(
            iteration,
            "DFG",
            load_for_inference,
            data,
            input_shape,
            num_classes,
            hidden_units,
            verbose,
            batch_size,
            epochs,
        )
        Vanilla = define_and_train(
            iteration,
            "Vanilla",
            load_for_inference,
            data,
            input_shape,
            num_classes,
            hidden_units,
            verbose,
            batch_size,
            epochs,
        )

        # test models
        for reliability_setting in reliability_settings:
            calc_accuracy(
                iteration,
                "ResiliNet",
                ResiliNet,
                ResiliNet_no_information_flow_map,
                reliability_setting,
                output_list,
                data,
            )
            calc_accuracy(
                iteration,
                "DFG",
                DFG,
                DFG_no_information_flow_map,
                reliability_setting,
                output_list,
                data,
            )
            calc_accuracy(
                iteration,
                "Vanilla",
                Vanilla,
                Vanilla_no_information_flow_map,
                reliability_setting,
                output_list,
                data,
            )

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

        output_list.append(
            str(reliability_setting)
            + " ResiliNet accuracy: "
            + str(ResiliNet_acc)
            + "\n"
        )
        output_list.append(
            str(reliability_setting) + " DFG accuracy: " + str(DFG_acc) + "\n"
        )
        output_list.append(
            str(reliability_setting) + " Vanilla accuracy: " + str(Vanilla_acc) + "\n"
        )

        print(str(reliability_setting), "ResiliNet accuracy:", ResiliNet_acc)
        print(str(reliability_setting), "DFG accuracy:", DFG_acc)
        print(str(reliability_setting), "Vanilla accuracy:", Vanilla_acc)

        ResiliNet_std = np.std(output["ResiliNet"][str(reliability_setting)], ddof=1)
        DFG_std = np.std(output["DFG"][str(reliability_setting)], ddof=1)
        Vanilla_std = np.std(output["Vanilla"][str(reliability_setting)], ddof=1)

        output_list.append(
            str(reliability_setting) + " ResiliNet std: " + str(ResiliNet_std) + "\n"
        )
        output_list.append(
            str(reliability_setting) + " DFG std: " + str(DFG_std) + "\n"
        )
        output_list.append(
            str(reliability_setting) + " Vanilla std: " + str(Vanilla_std) + "\n"
        )

        print(str(reliability_setting), "ResiliNet std:", ResiliNet_std)
        print(str(reliability_setting), "DFG std:", DFG_std)
        print(str(reliability_setting), "Vanilla std:", Vanilla_std)

    save_output(output_name, output_list)
    print(output)
