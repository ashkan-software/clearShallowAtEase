from keras.models import Sequential
from keras.layers import Dense,Input,Lambda, Activation
from keras.models import Model

PARTITION_SETING = 1 # ARCHITECTURE ONE: 1->2->3->4. ARCHITECTURE TWO: 2->3->2->3. 

def define_vanilla_model_MLP(num_vars,num_classes,hidden_units):
    # IoT Node (input image)
    img_input = Input(shape = (num_vars,))

    # edge node
    edge = define_MLP_architecture_edge(img_input, hidden_units)
    
    # fog node 2
    fog2 = Lambda(lambda x: x * 1,name="node3_input")(edge)
    fog2 = define_MLP_architecture_fog2(fog2, hidden_units)
    
    # fog node 1
    fog1 = Lambda(lambda x: x * 1,name="node2_input")(fog2)
    fog1 = define_MLP_architecture_fog1(fog1, hidden_units)
    
    # cloud node
    cloud = Lambda(lambda x: x * 1,name="node1_input")(fog1)
    cloud = define_MLP_architecture_cloud(cloud, hidden_units, num_classes)

    model = Model(inputs=img_input, outputs=cloud)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def define_MLP_architecture_edge(img_input,hidden_units):
    if PARTITION_SETING == 1:
        edge_output = Dense(units=hidden_units,name="edge_output_layer",activation='relu')(img_input)
    else: # PARTITION_SETING == 2
        edge = Dense(units=hidden_units,name="edge_input_layer",activation='relu')(img_input)
        edge_output = Dense(units=hidden_units,name="edge_output_layer",activation='relu')(edge)
    return edge_output

def define_MLP_architecture_fog2(fog2_input,hidden_units):
    if PARTITION_SETING == 1:
        fog2 = Dense(units=hidden_units,name="fog2_input_layer",activation='relu')(fog2_input)
        fog2_output = Dense(units=hidden_units,name="fog2_output_layer",activation='relu')(fog2)
    else: # PARTITION_SETING == 2
        fog2 = Dense(units=hidden_units,name="fog2_input_layer",activation='relu')(fog2_input)
        fog2 = Dense(units=hidden_units,name="fog2_layer_1",activation='relu')(fog2)
        fog2_output = Dense(units=hidden_units,name="fog2_output_layer",activation='relu')(fog2)
    return fog2_output

def define_MLP_architecture_fog1(fog1_input,hidden_units):
    if PARTITION_SETING == 1:
        fog1 = Dense(units=hidden_units,name="fog1_input_layer",activation='relu')(fog1_input)
        fog1 = Dense(units=hidden_units,name="fog1_layer_1",activation='relu')(fog1)
        fog1_output = Dense(units=hidden_units,name="fog1_output_layer",activation='relu')(fog1)
    else: # PARTITION_SETING == 2
        fog1 = Dense(units=hidden_units,name="fog1_input_layer",activation='relu')(fog1_input)
        fog1_output = Dense(units=hidden_units,name="fog1_output_layer",activation='relu')(fog1)
    return fog1_output

def define_MLP_architecture_cloud(cloud_input, hidden_units, num_classes):
    if PARTITION_SETING == 1:
        cloud = Dense(units=hidden_units,name="cloud_input_layer",activation='relu')(cloud_input)
        cloud = Dense(units=hidden_units,name="cloud_layer_1",activation='relu')(cloud)
        cloud = Dense(units=hidden_units,name="cloud_layer_2",activation='relu')(cloud)
        cloud = Dense(units=hidden_units,name="cloud_layer_3",activation='relu')(cloud)
    else: # PARTITION_SETING == 2
        cloud = Dense(units=hidden_units,name="cloud_input_layer",activation='relu')(cloud_input)
        cloud = Dense(units=hidden_units,name="cloud_layer_1",activation='relu')(cloud)
        cloud = Dense(units=hidden_units,name="cloud_layer_2",activation='relu')(cloud)
    cloud_output = Dense(units=num_classes,activation='softmax',name = "output")(cloud)
    return cloud_output