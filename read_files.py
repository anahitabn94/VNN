import argparse
import os
import numpy as np
import csv


def is_network(file_name):
    _, ext = os.path.splitext(file_name)
    if ext not in ['.h5']:
        raise argparse.ArgumentTypeError('only .h5 format is supported')
    return file_name


def is_dataset(file_name):
    _, ext = os.path.splitext(file_name)
    if ext not in ['.csv']:
        raise argparse.ArgumentTypeError('only .csv format is supported')
    return file_name


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value is expected.')


def load_data(path_data):
    with open(path_data) as file:
        rows = np.array([[int(float(i)) for i in row] for row in csv.reader(file)])
    y_test, x_test = rows[:, 0], rows[:, 1:]
    if "mnist" in path_data.lower():
        x_test = x_test / 255.0
    return x_test, y_test


def model_properties(model):
    model.compile(optimizer='sgd', loss='SparseCategoricalCrossentropy', metrics=['accuracy'])
    W_model = model.get_weights()
    model.summary()
    n_neu = dict()
    n_neu_cum = dict()
    tmp = np.prod(list(model.layers[0].input_shape)[1:])
    n_neu[0] = [tmp]
    n_neu_cum[0] = [tmp]
    W = dict()
    layer_type = dict()
    layer_activation = dict()
    k_layer = 1
    i_weight = 0
    for k in range(len(model.layers)):
        if k == len(model.layers) - 1 and model.layers[k].__class__.__name__ == 'Dense':  # last layer
            tmp = np.prod(list(model.layers[k].output_shape)[1:])
            W[k_layer] = [W_model[i_weight], W_model[i_weight + 1]]
            layer_type[k_layer] = 'Dense'
            layer_activation[k_layer] = 'none'
            n_neu[k_layer] = [tmp]
            n_neu_cum[k_layer] = [n_neu_cum[k_layer - 1][-1] + tmp]
        elif model.layers[k].__class__.__name__ == 'Dense':
            tmp = np.prod(list(model.layers[k].output_shape)[1:])
            W[k_layer] = [W_model[i_weight], W_model[i_weight + 1]]
            layer_type[k_layer] = 'Dense'
            layer_activation[k_layer] = model.layers[k].activation.__name__
            if layer_activation[k_layer] == 'none':
                n_neu[k_layer] = [tmp]
                n_neu_cum[k_layer] = [n_neu_cum[k_layer - 1][-1] + tmp]
            else:
                n_neu[k_layer] = [tmp, tmp]
                n_neu_cum[k_layer] = [n_neu_cum[k_layer - 1][-1] + tmp,
                                      n_neu_cum[k_layer - 1][-1] + np.sum(n_neu[k_layer])]
            i_weight += 2
            k_layer += 1
    return W, layer_type, layer_activation, n_neu, n_neu_cum
