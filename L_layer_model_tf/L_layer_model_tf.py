# -*- coding: utf-8 -*-
# This Program

import time
import math
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import h5py


def preprocess(X):
    # The "-1" makes reshape flatten the remaining dimensions
    X_flatten = X.reshape(X.shape[0], -1).T
    # Standardize data to have values between 0 and 1
    return X_flatten / 255.


def load_data(filename):
    """
    Dataset format should is h5 and look like this:
    datasets/
    --filename
    ----train_x (m, 64, 64, 3)
    ----train_y (m,)

    m := number of training examples
    filename := e.g. "mydataset.h5"
    """
    dataset = h5py.File(str(filename), "r")
    X = np.array(dataset["train_x"][:])
    Y = np.array(dataset["train_y"][:])
    # reshape from (m,) to (1, m)
    Y = Y.reshape((1, Y.shape[0]))
    return X, Y


def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, shape=[n_x, None], name="X")
    Y = tf.placeholder(tf.float32, shape=[n_y, None], name="Y")

    return X, Y


def initialize_parameters_binary():
    tf.set_random_seed(1)

    W1 = tf.get_variable("W1", [25, 12288], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [25, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [12, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [1, 12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [1, 1], initializer=tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters


def L_initialize_parameters_binary(layer_dims):
    tf.set_random_seed(1)

    L = len(layer_dims)
    parameters = {}
    for l in range(L-1):
        parameters["W" + str(l+1)] = tf.get_variable("W" + str(l+1), [layer_dims[l+1], layer_dims[l]], initializer=tf.contrib.layers.xavier_initializer(seed=1))
        parameters["b" + str(l+1)] = tf.get_variable("b" + str(l+1), [layer_dims[l+1], 1], initializer=tf.zeros_initializer())

    return parameters


def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)

    return Z3


def L_forward_propagation(X, parameters):
    Al = X
    L = int(len(parameters) / 2)
    for l in range(L):
        Zl = tf.add(tf.matmul(parameters["W" + str(l+1)], Al), parameters["b" + str(l+1)])
        Al = tf.nn.relu(Zl)

    return Zl


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    m = X.shape[1]
    mini_batches = []
    np.random.seed(seed)

    # shuffle
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

    # partition except end case
    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # end case
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def three_layer_model_tf(X_train, Y_train, X_test, Y_test, learning_rate=0.0001,
                     num_epochs=200, minibatch_size=32, print_cost=True):
    # to be able to rerun the model without overwriting tf variables
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 1
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []

    # create Placeholders
    X, Y = create_placeholders(n_x, n_y)

    # initialize parameters
    parameters = initialize_parameters_binary()

    # forward propagation
    Z3 = forward_propagation(X, parameters)

    # compute cost
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.transpose(Z3), labels=tf.transpose(Y)))

    # backpropagation
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # initialize all tf variables
    init = tf.global_variables_initializer()

    # start session to compute tensorflow graph
    with tf.Session() as sess:
        # run the initialization
        sess.run(init)

        # training loop
        for epoch in range(num_epochs):
            epoch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1  # new seed for each minibatch
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch

                # run tf graph on a single minibatch
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                epoch_cost += minibatch_cost / num_minibatches

            # print cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
        """
        # plot cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        """

        # get parameters from tf session
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        # softmax accuracy
        # correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        # Calculate the correct predictions
        # ZZZ = sess.run(Z3)
        #print('Z3: ' + str(tf.round(tf.sigmoid(Z3)).eval({X: X_train, Y: Y_train})))
        correct_prediction = tf.equal(tf.round(tf.sigmoid(Z3)), Y)

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters


def L_layer_model_tf(X_train, Y_train, X_test, Y_test, layer_dims, learning_rate=0.0001,
                     num_epochs=200, minibatch_size=32, print_cost=True):
    # to be able to rerun the model without overwriting tf variables
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 1
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []

    # create Placeholders
    X, Y = create_placeholders(n_x, n_y)

    # initialize parameters
    parameters = L_initialize_parameters_binary(layer_dims)

    # forward propagation
    Z3 = L_forward_propagation(X, parameters)

    # compute cost
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.transpose(Z3), labels=tf.transpose(Y)))

    # backpropagation
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # initialize all tf variables
    init = tf.global_variables_initializer()

    # start session to compute tensorflow graph
    with tf.Session() as sess:
        # run the initialization
        sess.run(init)

        # training loop
        for epoch in range(num_epochs):
            epoch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1  # new seed for each minibatch
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch

                # run tf graph on a single minibatch
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                epoch_cost += minibatch_cost / num_minibatches

            # print cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        """
        # plot cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        """

        # get parameters from tf session
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        # softmax accuracy
        # correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        # Calculate the correct predictions
        # ZZZ = sess.run(Z3)
        #print('Z3: ' + str(tf.round(tf.sigmoid(Z3)).eval({X: X_train, Y: Y_train})))
        correct_prediction = tf.equal(tf.round(tf.sigmoid(Z3)), Y)

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters


def main():
    print(time.time())

    filename = "../datasets/catvnoncat_2.h5"
    X_train, Y_train = load_data(filename)
    X_train = preprocess(X_train)

    test_filename = "../datasets/train_catvnoncat.h5"
    test_dataset = h5py.File(test_filename, "r")
    X_test = np.array(test_dataset["train_set_x"][:])
    Y_test = np.array(test_dataset["train_set_y"][:])
    # reshape from (m,) to (1, m)
    Y_test = Y_test.reshape((1, Y_test.shape[0]))
    X_test = preprocess(X_test)

    layer_dims = [12288, 25, 25, 12, 1]
    learning_rate = 0.0001
    num_epochs = 400
    minibatch_size = 64
    print_cost = True

    start = time.time()
    print("Start time: ", start)
    # execute model
    params = L_layer_model_tf(X_train, Y_train, X_test, Y_test, layer_dims, learning_rate,
                              num_epochs, minibatch_size, print_cost)
    diff = time.time() - start
    print("Time: ", diff)


if __name__ == "__main__":
    main()
#
#
#
#
#
#
#
#
#
#
