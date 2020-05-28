from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import warnings

import keras.backend as K
import keras.layers as layers
from keras_applications.imagenet_utils import _obtain_input_shape, get_submodules_from_kwargs
import random 
from Experiment.common_exp_methods_CNN import set_hyperconnection_weights, define_hyperconnection_weight_lambda_layers
from Experiment.cnn_Vanilla_MobileNet import define_cnn_architecture_IoT, define_cnn_architecture_cloud, define_cnn_architecture_edge, define_cnn_architecture_fog, imagenet_related_functions
from Experiment.common_exp_methods import compile_keras_parallel_model

default_skip_hyperconnection_config = [1,1]
def define_deepFogGuard_CNN_MobileNet(input_shape=None,
                                    alpha=1.0,
                                    depth_multiplier=1,
                                    include_top=True,
                                    input_tensor=None,
                                    pooling=None,
                                    classes=1000,
                                    strides = (2,2),
                                    skip_hyperconnection_config = default_skip_hyperconnection_config, # binary representating if a skip hyperconnection is alive [e1,IoT]
                                    reliability_setting=[1.0,1.0], # reliability of a node between 0 and 1 [f1,e1]
                                    hyperconnection_weights_scheme = 1,
                                    num_gpus = 1,
                                    weights=None,
                                    **kwargs):

    if weights == 'imagenet':
        weights = None
        imagenet_related_functions(weights, input_shape, include_top, classes, depth_multiplier, alpha)

    hyperconnection_weight_IoTe, hyperconnection_weight_IoTf,hyperconnection_weight_ef,hyperconnection_weight_ec,hyperconnection_weight_fc = set_hyperconnection_weights(
        hyperconnection_weights_scheme, 
        reliability_setting, 
        skip_hyperconnection_config)
    multiply_hyperconnection_weight_layer_IoTe, multiply_hyperconnection_weight_layer_IoTf, multiply_hyperconnection_weight_layer_ef, multiply_hyperconnection_weight_layer_ec, multiply_hyperconnection_weight_layer_fc = define_hyperconnection_weight_lambda_layers(
        hyperconnection_weight_IoTe,
        hyperconnection_weight_IoTf,
        hyperconnection_weight_ef,
        hyperconnection_weight_ec,
        hyperconnection_weight_fc)

    # Determine proper input shape and default size.
    img_input = layers.Input(shape=input_shape)  

    # iot node
    iot_output,skip_iotfog = define_cnn_deepFogGuard_architecture_IoT(input_shape,alpha,img_input, strides = strides)

    # edge node
    edge_output, skip_edgecloud = define_cnn_deepFogGuard_architecture_edge(iot_output,alpha, depth_multiplier, multiply_hyperconnection_weight_layer_IoTe, strides = strides)

    # fog node
    fog_output = define_cnn_deepFogGuard_architecture_fog(skip_iotfog, edge_output, alpha, depth_multiplier, multiply_hyperconnection_weight_layer_IoTf, multiply_hyperconnection_weight_layer_ef, strides = strides)

    # cloud node
    cloud_output = define_cnn_deepFogGuard_architecture_cloud(fog_output, skip_edgecloud, alpha, depth_multiplier, classes, include_top, pooling, multiply_hyperconnection_weight_layer_fc, multiply_hyperconnection_weight_layer_ec)

    model, parallel_model = compile_keras_parallel_model(img_input, cloud_output, num_gpus)
    return model, parallel_model

def define_cnn_deepFogGuard_architecture_IoT(input_shape, alpha, img_input, strides = (2,2)):
    iot_output = define_cnn_architecture_IoT(img_input,alpha,strides = strides)

    # Need to go from (32,32,3) to (32,32,64)
    # 1x1 conv2d is used to change the filter size (from 3 to 64).  
    # cifar-10
    if strides == (1,1):
        # 64 (alpha=0.5), 96 (alpha=0.75)
        skip_iotfog = layers.Conv2D(96,(1,1),strides = 1, use_bias = False, name = "skip_hyperconnection_iotfog")(iot_output)
    elif strides == (2,2):
        skip_iotfog = layers.Conv2D(96,(1,1),strides = 2, use_bias = False, name = "skip_hyperconnection_iotfog")(iot_output)
    else:
        raise ValueError("Invalid stride configuration")
    return iot_output, skip_iotfog

def define_cnn_deepFogGuard_architecture_edge(iot_output, alpha, depth_multiplier, multiply_hyperconnection_weight_layer_IoTe = None, strides = (2,2), edge_failure_lambda = None):
    if multiply_hyperconnection_weight_layer_IoTe != None:
        iot_output = multiply_hyperconnection_weight_layer_IoTe(iot_output)
    edge_output = define_cnn_architecture_edge(iot_output,alpha,depth_multiplier, strides= strides)
    if edge_failure_lambda != None:
         edge_output = edge_failure_lambda(edge_output)
    # used stride 4 to match (31,31,64) to (7,7,256)
    # 1x1 conv2d is used to change the filter size (from 64 to 256).  Stride is 4 for 31->7
    # 256 (alpha=0.5), 384 (alpha=0.75)
    skip_edgecloud = layers.Conv2D(384,(1,1),strides = 4, use_bias = False, name = "skip_hyperconnection_edgecloud")(edge_output)
    return edge_output, skip_edgecloud
   

def define_cnn_deepFogGuard_architecture_fog(skip_iotfog, edge_output, alpha, depth_multiplier, multiply_hyperconnection_weight_layer_IoTf = None, multiply_hyperconnection_weight_layer_ef = None, strides = (2,2)):
    if multiply_hyperconnection_weight_layer_IoTf == None or multiply_hyperconnection_weight_layer_ef == None:
        fog_input = layers.add([skip_iotfog, edge_output], name = "node2_input")
    else:
        fog_input = layers.add([multiply_hyperconnection_weight_layer_IoTf(skip_iotfog), multiply_hyperconnection_weight_layer_ef(edge_output)], name = "node2_input")
    fog = define_cnn_architecture_fog(fog_input,alpha,depth_multiplier)
    # cnn for imagenet does not need padding
    if strides == (2,2):
        fog_output = fog
    elif strides == (1,1):
        # pad from (7,7,256) to (8,8,256)
        fog_output = layers.ZeroPadding2D(padding = ((0, 1), (0, 1)), name = "fogcloud_connection_padding")(fog)
    else:
        raise ValueError("Incorrect stride value")
    
    return fog_output

def define_cnn_deepFogGuard_architecture_cloud(fog_output, skip_edgecloud, alpha, depth_multiplier, classes, include_top, pooling, multiply_hyperconnection_weight_layer_fc = None, multiply_hyperconnection_weight_layer_ec = None):
    if multiply_hyperconnection_weight_layer_fc == None or multiply_hyperconnection_weight_layer_ec == None:
        cloud_input = layers.add([fog_output, skip_edgecloud], name = "node1_input")
    else:
        cloud_input = layers.add([multiply_hyperconnection_weight_layer_fc(fog_output), multiply_hyperconnection_weight_layer_ec(skip_edgecloud)], name = "node1_input")
    cloud_output = define_cnn_architecture_cloud(cloud_input,alpha,depth_multiplier,classes,include_top,pooling)
    return cloud_output