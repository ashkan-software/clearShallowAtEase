import os

from keras.applications.imagenet_utils import preprocess_input
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import multi_gpu_model


num_train_examples = 1300000
num_test_examples = 50000
input_shape = (160, 160, 3)
alpha = 0.75
num_iterations = 1
# need to change this to be accurate
reliability_settings = [[1, 1], [0.98, 0.96], [0.95, 0.90], [0.85, 0.80]]
num_classes = 1000
epochs = 10
num_gpus = 4
num_workers = 32
strides = (2, 2)


def init_data(num_gpus):
    train_dir = "/path/to/train/dataset"
    test_dir = "/path/to/test/dataset"
    input_shape = (160, 160)
    batch_size = 1024
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        # preprocessing_function=preprocess_input
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    train_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=input_shape,
        batch_size=batch_size * num_gpus,
        class_mode="sparse",
        shuffle=True,
        seed=42,
    )
    test_generator = test_datagen.flow_from_directory(
        # shuffle = False,
        directory=test_dir,
        target_size=input_shape,
        batch_size=batch_size * num_gpus,
        class_mode="sparse",
        seed=42,
    )
    return train_generator, test_generator


def get_model_weights_CNN_imagenet(
    model,
    parallel_model,
    model_name,
    load_for_inference,
    continue_training,
    model_file,
    train_generator,
    val_generator,
    num_train_examples,
    epochs,
    num_gpus,
    num_workers,
):
    if load_for_inference:
        parallel_model.load_weights(model_file)
        model = parallel_model.layers[-2]
        return model
    else:
        print(model_name)
        verbose = 1
        if num_gpus > 1:
            if continue_training:
                parallel_model.load_weights(model_file)
                model = parallel_model.layers[-2]
            modelCheckPoint = ModelCheckpoint(model_file, save_weights_only=True)
            parallel_model.fit_generator(
                generator=train_generator,
                steps_per_epoch=num_train_examples / train_generator.batch_size,
                epochs=epochs,
                workers=num_workers,
                class_weight=None,
                verbose=verbose,
                callbacks=[modelCheckPoint],
            )
            # save the weights
            parallel_model.save_weights(model_file)
            return model
        else:
            model.fit_generator(
                generator=train_generator,
                steps_per_epoch=num_train_examples / train_generator.batch_size,
                epochs=epochs,
                workers=num_workers,
                class_weight=None,
                verbose=verbose,
            )
            # save the weights
            model.save_weights(model_file)
            return model
