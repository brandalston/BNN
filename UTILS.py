"""
This code is used for evaluating the test performance of the BNNs found by each approach.
The "BinarizedNetwork" class creates the bnn, imports the weights given by a model,
and evaluate the network performance over a dataset.
This class is also used by the HA methods to compute the layer activations.
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from dataload import mnist_test_numpy
import os, math, warnings, csv
warnings.filterwarnings("ignore")

example_skip_dict = {'s_138': 20, 's_15': 15, 's_89': 10, 's_42': 5, 's_0': 0}


class BinarizedNetwork:

    def __init__(self, neurons_per_layer):
        """
        "architecture" is a list of numbers indicating how many neurons each layer has
          e.g. [2,2,1] -> 2 input neurons, then 2 neurons on a hidden layer, and one output neuron
        """
        self.neurons_per_layer = neurons_per_layer
        self._create_network(neurons_per_layer)

    def _create_network(self, neurons_per_layer):
        self.sess = tf.Session()
        n_inputs = neurons_per_layer[0]
        n_outputs = neurons_per_layer[-1]
        self.imgs = tf.placeholder(tf.float32, shape=[None, n_inputs])
        self.labels = tf.placeholder(tf.float32, shape=[None, n_outputs])
        self.weight_values = tf.placeholder(tf.float32)
        self.bias_values = tf.placeholder(tf.float32)

        # list of operators to update weights on a given layer
        self.update_weights = []
        self.patterns = [self.imgs]
        self.weights = []
        self.biases = []

        # Adding hidden layers
        x = self.imgs
        for layer_id in range(1, len(neurons_per_layer)):
            x_prev = x

            # Computing hidden patterns
            fc_shape = [neurons_per_layer[layer_id - 1], neurons_per_layer[layer_id]]

            n_outputs = fc_shape[1]
            W = weight_variable(fc_shape)
            b = bias_variable([n_outputs])
            a = tf.matmul(x_prev, W) + b

            # zero is considered a positive activation by default
            x = tf.sign(tf.sign(a) + 0.1)

            # Set weights from MIP
            w_copy = tf.assign(W, tf.reshape(self.weight_values, fc_shape))
            b_copy = tf.assign(b, tf.reshape(self.bias_values, [n_outputs]))
            self.update_weights.append([w_copy, b_copy])

            # Set weights and biases from this layer
            # (this allows for computing the costs of each pattern)
            self.weights.append(W)
            self.biases.append(b)

            # Get activations in the current layer
            self.patterns.append(x)

        # Computing accuracy
        score = tf.multiply(self.labels, x)
        score = tf.reduce_sum(score, 1) >= n_outputs
        self.performance = tf.reduce_mean(tf.cast(score, tf.float32))

        # initialize variables
        self.sess.run(tf.global_variables_initializer())

    def get_activations(self, imgs):
        return self.sess.run(self.patterns, {self.imgs: imgs})

    def get_number_of_weights(self):
        total = 0
        for i in range(1, len(self.neurons_per_layer)):
            weights, biases = self.sess.run([self.weights[i - 1], self.biases[i - 1]], {})
            total += np.sum(np.abs(weights)) + np.sum(np.abs(biases))
        return total

    def test_network(self, imgs, labels):
        """
        The "all-good" evaluation metric:
            A multiclass instance is considered to be correctly classified iff
            its one-hot embedding is perfectly predicted by the network.
            Aka perfect classifier would have 0.0 performance for a 10-class problem
        """
        return self.sess.run(self.performance, {self.imgs: imgs, self.labels: labels})

    def update_layer(self, layer_id, weights, biases):
        self.sess.run(self.update_weights[layer_id], {self.weight_values: weights, self.bias_values: biases})

    def close(self):
        self.sess.close()


def weight_variable(shape):
    initial = tf.random_uniform(shape, minval=-1, maxval=2, dtype=tf.int32)
    initial = tf.cast(initial, tf.float32)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.random_uniform(shape, minval=-1, maxval=2, dtype=tf.int32)
    initial = tf.cast(initial, tf.float32)
    return tf.Variable(initial)


def get_one_hot_encoding(a, n_classes=10):
    b = np.zeros((a.size, n_classes))
    b[np.arange(a.size), a] = 1
    return b


def model_acc(net, weights, biases, images, labels):
    bnn = BinarizedNetwork(net)
    for i in range(len(weights)):
        bnn.update_layer(i, weights[i], biases[i])
    train_performance = bnn.test_network(images, labels)
    images, labels = mnist_test_numpy()
    # mapping labels to -1/1 vectors
    labels = 2.0*get_one_hot_encoding(labels) - 1.0
    test_performance = bnn.test_network(images, labels)
    # print("Test performance = %0.2f"%test_performance)

    # clossing the network sessions
    bnn.close()

    return train_performance, test_performance


def model_summary(results_dict, file_out=None):
    print()
    if 'GD' not in results_dict['obj_func']:
        with open(file_out, mode='a') as results:
            results_writer = csv.writer(results, delimiter=',', quotechar='"')
            results_writer.writerow([
                results_dict['rand_state'], results_dict['train_size'],
                results_dict["train_acc"], results_dict["test_acc"], results_dict["run_time"],
                results_dict['n_hidden_layers'], results_dict['obj_func'],
                results_dict["learning_rate"], results_dict["tf_seed"], results_dict['TL'],
                results_dict['MIPGap'], results_dict['ObjVal'], results_dict['ObjBound'],
            ])
            results.close()
    else:
        with open(file_out, mode='a') as results:
            results_writer = csv.writer(results, delimiter=',', quotechar='"')
            results_writer.writerow([
                results_dict['rand_state'], results_dict['train_size'],
                results_dict["train_acc"], results_dict["test_acc"], results_dict["run_time"],
                results_dict['n_hidden_layers'], results_dict["obj_func"],
                results_dict["learning_rate"], results_dict["tf_seed"], results_dict['TL']])
            results.close()
