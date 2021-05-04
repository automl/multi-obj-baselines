# Bag of Baselines

Bag of Baselines implements several multi-objective opimisation methods to create a performance benchmark on two small datasets. To learn more about this work, check out the [publication](https://arxiv.org/abs/2105.01015).

### Methods

The following methods are proposed and implemented:

1. **SH-EMOA**: Speeding up Evolutionary Multi-Objective Algorithms

2. **MO-BOHB**: Generalization of BOHB to an Arbitrary Number of Objectives

3. **MS-EHVI**: Mixed Surrogate Expected Hypervolume Improvement

4. **MO-BANANAS**

5. **BULK & CUT**

### Datasets

Performance of the methods was evaluated using the following datasets: [Oxford-Flowers dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/) and [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist).


### Organization

 * The specific code for each of the methods (the main logic of each algorithm) is stored in the [methods](https://github.com/automl/multi-obj-baselines/tree/main/baselines/methods) folder.

 * In the [examples](https://github.com/automl/multi-obj-baselines/tree/main/examples) folder you will find a small Python script to run each of the available methods (for the "Fashion-MNIST" or the "flowers" dataset).

 * Code defining the search space and the evaluation function of the two different problems are defined in the [problems](https://github.com/automl/multi-obj-baselines/tree/main/baselines/problems) folder. 

