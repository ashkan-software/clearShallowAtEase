
# [Code] Failout: Achieving Failure-Resilient Inference in Distributed Neural Networks

This repository is for the code related to the paper *Failout: Achieving Failure-Resilient Inference in Distributed Neural Networks*.

## What is Failout?

When a neural network is partitioned and distributed across physical nodes, failure of physical nodes causes the failure of the neural units that are placed on those nodes, which results in a significant performance drop. We introduce ResiliNet, a scheme for making inference in distributed neural networks resilient to physical node failures. ResiliNet combines two concepts to provide resiliency: skip connection in residual neural networks, and a novel technique called failout. Failout simulates physical node failure conditions during training using dropout, and is specifically designed to improve the resiliency of distributed neural networks. 

## How to run

All of the files to run the experiments reside in `Experiment` folder. The name of the python files start with the *experiment dataset*, followed by the *type of experiment*. For instance, you can run the *Health* experiment, and get the *average accuracy* by running:

```
 python Experiment/health_average_accuracy.py 
```

If you are using Python 3.x, replace `python` with `python3`. In general, you can run the experiments using the following rule:

```
 python Experiment/<dataset>_<experiment-type>.py 
```
  
 where `<dataset>` is either `health`, `cifar`, or `imagenet` (Note that, `camera` is only for *experimental* purposes with a distribtued neural network that is both vertically and horizontally split.), and `<experiment-type>` is either `average_accuracy`, `hyperconnection_weight`, `failout_rate`, or `skiphyperconnection_sensitivity`. 

The datasets and the preprocessing methods are explained in the paper. The experiments are as follows:

- `average_accuracy`: (Section 3.3). Obtains average accuracy, in addition to accuracy for individual physical node failures.
- `hyperconnection_weight`: (Section 3.4.1) Obtains results for different choices of hyperconnection weights.
- `failout_rate`: (Section 3.4.2) Obtains results for different rates of failout.
``skiphyperconnection_sensitivity`: (Section 3.4.3) Obtains results regarding which skip hyperconnections are more critical. 

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


