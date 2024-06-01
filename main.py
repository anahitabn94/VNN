from pytictoc import TicToc
from read_files import *
from help_funcs import *
import tensorflow as tf
import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='VNN')
    parser.add_argument('--path_network', type=is_network, default="Models/mnist_relu_3_50.h5", help='name of the NN')
    parser.add_argument('--path_dataset', type=is_dataset, default="Dataset/mnist_validation.csv", help='dataset')
    parser.add_argument('--epsilon', type=float, default=0.0, help='degree of freedom')
    args = parser.parse_args()

    path_network = args.path_network
    path_dataset = args.path_dataset
    epsilon = args.epsilon

    print(path_dataset)
    print(path_network)

    model = tf.keras.models.load_model(path_network)
    W, layer_type, layer_activation, n_neu, n_neu_cum = model_properties(model)
    data, labels = load_data(path_dataset)

    score = model.evaluate(data, labels, verbose=0)
    print("The accuracy of this neural network model is ", score[1] * 100, "%")

    predictions = np.squeeze(model.predict(data))
    numTrue = 0
    gb_model = {l + 1: None for l in range(sum(1 for v in layer_activation.values() if v == 'relu'))}
    ii = -1
    t_all = TicToc()
    t_all.tic()
    for i, (prediction, label) in enumerate(zip(predictions, labels)):
        lyr = 0
        if label == np.argmax(prediction):
            ii += 1
            print('Sample number is ', i + 1)
            numTrue += 1
            center = {0: data[i][..., np.newaxis]}
            oas = dict()
            center, oas, gb_inds = net_propagate(1, W, layer_type, layer_activation, center, oas)
            for j in range(len(layer_type)):
                if layer_type[j + 1] == 'Dense':
                    lyr += 1
                    if layer_activation[j + 1] == 'relu':
                        if layer_type[j + 1] == 'Dense':
                            gb_model[lyr] = model_generator(ii, W[j + 1], {j: center[j],
                                                            j + 1: np.expand_dims(center[j + 1][:, 0], axis=-1)},
                                                            layer_type[j + 1], gb_inds[j + 1], gb_model[lyr], epsilon)

    update_model_weights(layer_activation, layer_type, gb_model, weight_opt, W, model)
    print(model.evaluate(data, labels, verbose=0))

    filename = os.path.basename(path_network)
    root, ext = os.path.splitext(filename)
    new_filename = root + f"_VNN_epsilon{epsilon}" + ext
    output_path = os.path.join("VNN", new_filename)
    model.save(output_path)

    print("Processing time is ", t_all.tocvalue(), " seconds")
