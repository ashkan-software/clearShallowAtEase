from Experiment.cnn_Vanilla_MobileNet import define_vanilla_CNN_MobileNet
from Experiment.cnn_deepFogGuard_MobileNet import define_deepFogGuard_CNN_MobileNet
from Experiment.cnn_ResiliNet_MobileNet import define_ResiliNet_CNN_MobileNet
from Experiment.cnn_Vanilla_ResNet import define_vanilla_CNN_ResNet
from Experiment.cnn_deepFogGuard_ResNet import define_deepFogGuard_CNN_ResNet
from Experiment.cnn_ResiliNet_ResNet import define_ResiliNet_CNN_ResNet
from Experiment.cnn_ResiliNet_ResNet import MUX_ADDS as MUX_ADDS_ResNet
from Experiment.cnn_ResiliNet_MobileNet import MUX_ADDS as MUX_ADDS_MobileNet

def define_model(iteration, model_name, dataset_name, input_shape, classes, alpha, strides, num_gpus, weights):
    # ResiliNet
    if model_name == "ResiliNet":
        if dataset_name == "cifar_resnet":
            mux_adds_str = "mux_adds" if MUX_ADDS_ResNet else "" 
            model, parallel_model = define_ResiliNet_CNN_ResNet(input_shape=input_shape, classes=classes, block='basic', residual_unit='v2',
                                repetitions=[2, 2, 2, 2], initial_filters=64, activation='softmax', include_top=True,
                                input_tensor=None, dropout=None, transition_dilation_rate=(1, 1),
                                initial_strides=(2, 2), initial_kernel_size=(7, 7), initial_pooling='max',
                                final_pooling=None, top='classification',
                                num_gpus = num_gpus)
        else:
            mux_adds_str = "mux_adds" if MUX_ADDS_MobileNet else "" 
            model, parallel_model = define_ResiliNet_CNN_MobileNet(classes=classes,input_shape = input_shape,alpha = alpha, strides = strides, num_gpus=num_gpus, weights=weights)
        model_file = "models/" + dataset_name + str(iteration) +mux_adds_str + 'average_accuracy_ResiliNet.h5'
    # deepFogGuard
    if model_name == "deepFogGuard":
        if dataset_name == "cifar_resnet":
            model, parallel_model = define_deepFogGuard_CNN_ResNet(input_shape=input_shape, classes=classes, block='basic', residual_unit='v2',
                                    repetitions=[2, 2, 2, 2], initial_filters=64, activation='softmax', include_top=True,
                                    input_tensor=None, dropout=None, transition_dilation_rate=(1, 1),
                                    initial_strides=(2, 2), initial_kernel_size=(7, 7), initial_pooling='max',
                                    final_pooling=None, top='classification', num_gpus = num_gpus)
        else:
            model, parallel_model = define_deepFogGuard_CNN_MobileNet(classes=classes,input_shape = input_shape,alpha = alpha, strides = strides, num_gpus=num_gpus, weights=weights)
        model_file =  "models/"+ dataset_name  + str(iteration) + 'average_accuracy_deepFogGuard.h5'
    # Vanilla model
    if model_name == "Vanilla":
        if dataset_name == "cifar_resnet":
            model, parallel_model = define_vanilla_CNN_ResNet(input_shape=input_shape, classes=classes, block='basic', residual_unit='v2',
                            repetitions=[2, 2, 2, 2], initial_filters=64, activation='softmax', include_top=True,
                            input_tensor=None, dropout=None, transition_dilation_rate=(1, 1),
                            initial_strides=(2, 2), initial_kernel_size=(7, 7), initial_pooling='max',
                            final_pooling=None, top='classification', num_gpus = num_gpus)
        else:
            model, parallel_model = define_vanilla_CNN_MobileNet(classes=classes,input_shape = input_shape,alpha = alpha, strides = strides, num_gpus=num_gpus, weights=weights)
        model_file = "models/" + dataset_name  + str(iteration) + 'average_accuracy_vanilla.h5'
    
    return model, parallel_model, model_file