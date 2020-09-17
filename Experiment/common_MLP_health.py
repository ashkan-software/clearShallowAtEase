import os

import numpy as np
from Experiment.data import ProcessedData
from Experiment.data_handler_health import load_data
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split


num_classes = 13
reliability_settings = [
    [1, 1, 1],
    [0.99, 0.96, 0.92],
    [0.95, 0.91, 0.87],
    [0.85, 0.8, 0.78],
]
num_train_epochs = 50
hidden_units = 250
batch_size = 1024
num_iterations = 10


def init_data():
    if not os.path.exists("models/"):
        os.mkdir("models/")
    data, labels = load_data("mHealth_complete.log")
    # split data into train, val, and test
    # 80/10/10 split
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, random_state=42, test_size=0.20, shuffle=True
    )
    val_data, test_data, val_labels, test_labels = train_test_split(
        test_data, test_labels, random_state=42, test_size=0.50, shuffle=True
    )
    num_vars = len(train_data[0])
    return ProcessedData(
        train_data,
        val_data,
        test_data,
        train_labels,
        val_labels,
        test_labels,
    ), num_vars


def get_model_weights_MLP_health(
    model,
    model_name,
    load_for_inference,
    model_file,
    training_data,
    training_labels,
    val_data,
    val_labels,
    num_train_epochs,
    batch_size,
    verbose,
):
    if load_for_inference:
        model.load_weights(model_file)
    else:
        print(model_name)
        modelCheckPoint = ModelCheckpoint(
            model_file,
            monitor="val_acc",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode="auto",
            period=1,
        )
        model.fit(
            x=training_data,
            y=training_labels,
            batch_size=batch_size,
            validation_data=(val_data, val_labels),
            callbacks=[modelCheckPoint],
            verbose=verbose,
            epochs=num_train_epochs,
            shuffle=True,
        )
        # load weights from epoch with the highest val acc
        model.load_weights(model_file)
