
# [Code] ResiliNet: Failure-Resilient Inference in Distributed Neural Networks

This repository is for the code related to the paper *ResiliNet: Failure-Resilient Inference in Distributed Neural Networks*.

## What is ResiliNet?

Federated Learning aims to train distributed deep models without sharing the raw data with the centralized server. Similarly, in Split Learning, by partitioning a neural network and distributing it across several physical nodes, activations and gradients are exchanged between physical nodes, rather than raw data. Nevertheless, when a neural network is partitioned and distributed among physical nodes, failure of physical nodes causes the failure of the neural units that are placed on those nodes, which results in a significant performance drop. 

ResiliNet is a scheme for making inference in distributed neural networks resilient to physical node failures. ResiliNet combines two concepts to provide resiliency: skip hyperconnection, a concept for skipping nodes in distributed neural networks similar to skip connection in resnets, and a novel technique called failout. Failout simulates physical node failure conditions during training using dropout, and is specifically designed to improve the resiliency of distributed neural networks. 

## How to run

All of the files to run the experiments reside in `Experiment` folder. The name of the python files start with the *experiment dataset*, followed by the *type of experiment*. For instance, you can run the *Health* experiment, and get the *average accuracy* by running:

```
 python Experiment/health_average_accuracy.py 
```

If you are using Python 3.x, replace `python` with `python3`. In general, you can run the experiments using the following rule:

```
 python Experiment/<dataset>_<experiment-type>.py 
```
  
 where `<dataset>` is either `health`, `cifar_mobilenet`,  `cifar_resnet`, or `imagenet` (Note that, `camera` is only for *experimental* purposes with a distribtued neural network that is both vertically and horizontally split), and `<experiment-type>` is either `average_accuracy`, `hyperconnection_weight`, `failout_rate`, or `skiphyperconnection_sensitivity`. 

The datasets and the preprocessing methods are explained in the paper. The experiments are as follows:

- `average_accuracy`: Obtains average accuracy, in addition to the accuracy for individual physical node failures.
- `hyperconnection_weight`: Obtains results for different choices of hyperconnection weights. (What is the best choice of weights for the hyperconnections?)
- `failout_rate`: Obtains results for different rates of failout. (What is the optimal rate of failout?)
``skiphyperconnection_sensitivity`: Obtains results for sensitivity of skip hyperconnections. (Which skip hyperconnections are more important?) 

In order to run the experiments for different partition of distributed DNN, you can choose between different values of `PARTITION_SETING`. The parameter `PARTITION_SETING`, depending on the experiment, is in the following files: `mlp_Vanilla_health.py`, `cnn_Vanilla_MobileNet.py`, and `cnn_Vanilla_ResNet.py`.

In order to run experiments using ResiliNet+, you can set `ResiliNetPlus` to `True`. The parameter `ResiliNetPlus`, depending on the experiment, is in the following files: `mlp_ResiliNet_health.py`, `cnn_ResiliNet_MobileNet.py`, and `cnn_ResiliNet_ResNet.py`.

## Dependencies

The following python packages are required to run these experiments. 
- Keras
- sklearn
- networkx
- pandas
- cv2

Some of these packages are required only for a particular experiment, so you may simply try running the experiment and see if any package is missing:


## Output

Once you run an experiments, you will see the output in the console (e.g. accuracy). When the experiment finished running, new folders will be created. These folders keep the results and the models associated with each experiment. 

`/results` keeps all of the result text and log files from training.

`/models` keeps all of the saved models after training.


