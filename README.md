# Verification-friendly Neural Network (VNN)

We introduce a post-training optimization framework designed to strike a balance between maintaining prediction performance and enhancing verification capabilities. Through our proposed approach, we develop Verification-friendly Neural Networks (VNNs) that exhibit prediction performance comparable to original Deep Neural Networks (DNNs), while also being compatible with formal verification techniques. This facilitates the establishment of robustness for a greater number of VNNs compared to their DNN counterparts, in a time-efficient manner.

This framework gets a neural network model and a dataset, and a value of $\epsilon$, which shows degree of freedom. The dataset should be a csv file consisting of 1-D signals as inputs and labels. The model should be a classifier in h5 format containing dense layers with ReLU activation functions for all the hidden layers. $\epsilon$ can be any float value from 0.0 to 0.99.

## Requirements 

Gurobi's Python Interface, Python 3.9 or higher, TensorFlow 2.10 or higher.


## Usage
```python
python main.py --path_network <path to the model> --path_dataset <path to the dataset> --epsilon <float between 0.0 and 0.99>

```

### Example

```python
python main.py --path_network Models/mnist_relu_2_100.h5 --path_dataset Dataset/mnist_validation.csv --epsilon 0.0

