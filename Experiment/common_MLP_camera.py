import os

import numpy as np
from Experiment.data import ProcessedData
from Experiment.data_handler_camera import load_dataset
from keras.callbacks import ModelCheckpoint
from sklearn.utils import class_weight


input_shape = (32, 32, 3)
# need to change this to be accurate
reliability_settings = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [0.99, 0.99, 0.99, 0.99, 0.98, 0.98, 0.98, 0.98],
    [0.95, 0.95, 0.9, 0.9, 0.8, 0.8, 0.75, 0.75],
    [0.8, 0.8, 0.75, 0.75, 0.7, 0.7, 0.65, 0.65],
]
num_classes = 3
hidden_units = 32
batch_size = 64
epochs = 50
num_iterations = 10


def init_data():
    if not os.path.exists("models/"):
        os.mkdir("models/")
    train_dir = "multiview-dataset/train_dir"
    val_dir = "multiview-dataset/test_dir"
    test_dir = "multiview-dataset/holdout_dir"
    img_size = (32, 32, 3)
    classes = ["person_images", "car_images", "bus_images"]
    training_data, training_labels, _, _ = load_dataset(train_dir, img_size, classes)
    val_data, val_labels, _, _ = load_dataset(val_dir, img_size, classes)
    test_data, test_labels, _, _ = load_dataset(test_dir, img_size, classes)

    training_data = np.array(training_data)
    val_data = np.array(val_data)
    test_data = np.array(test_data)

    # convert one-hot to integer encoding
    training_labels = np.array([np.where(r == 1)[0][0] for r in training_labels])
    val_labels = np.array([np.where(r == 1)[0][0] for r in val_labels])
    test_labels = np.array([np.where(r == 1)[0][0] for r in test_labels])
    # format images correctly to be used for MLP
    training_data = [
        training_data[:, 0],
        training_data[:, 1],
        training_data[:, 2],
        training_data[:, 3],
        training_data[:, 4],
        training_data[:, 5],
    ]
    val_data = [
        val_data[:, 0],
        val_data[:, 1],
        val_data[:, 2],
        val_data[:, 3],
        val_data[:, 4],
        val_data[:, 5],
    ]
    test_data = [
        test_data[:, 0],
        test_data[:, 1],
        test_data[:, 2],
        test_data[:, 3],
        test_data[:, 4],
        test_data[:, 5],
    ]
    return ProcessedData(training_data, val_data, test_data, training_labels, val_labels, test_labels)


def get_model_weights_MLP_camera(
    model,
    model_name,
    load_for_inference,
    model_file,
    data,
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
        class_weights = class_weight.compute_class_weight(
            "balanced", np.unique(data.train), data.train_labels
        )
        model.fit(
            x=data.train,
            y=data.train_labels,
            batch_size=batch_size,
            validation_data=(data.val, data.val_labels),
            callbacks=[modelCheckPoint],
            verbose=verbose,
            epochs=num_train_epochs,
            shuffle=True,
            class_weight=class_weights,
        )
        # load weights from epoch with the highest val acc
        model.load_weights(model_file)
