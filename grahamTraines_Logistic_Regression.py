__author__ = 'graham'
# Graham Traines
# CSCI 7090 Machine Learning
# Logistic Regression with Gradient Descent
# Demonstration

import numpy as np
import scipy as sp
import matplotlib.pyplot as mpl

"""
GO TO THE BOTTOM OF THE SCRIPT TO FIND THE START
"""

# method:
# ---Load data
def load_data_set(file_location, delimiter, column_y, column_x1):
    """
    :param file_location: string for data file location
    :param delimiter: ',' for CSV, etc.
    :param column_y: the column containing the target values
    :param column_x1: input data column -- right now, I only take 1
    TODO: add a parameter that is a set so I can take multiple input columns
    :return: Numpy Array of target values, input values, and bias term (bias term always = 1.0)
    """
    data = sp.genfromtxt(file_location, delimiter=delimiter, dtype=None)

    # Need to get everything after the headers

    X = data[1:, column_x1]
    Y = data[1:, column_y]

    # we make the cases 1 and -1 to fit with the Likelihood Formula
    # P(y | x) -> h(x) for y = +1
    # P(y | x) -> 1 - h(x) for y = -1
    y_numeric = [1.0 if entry == 'Yes' else -1.0 for entry in Y]

    # Will use this for x0
    ones = [1.0 for x in X]

    return np.array(zip(y_numeric, X, ones), dtype='float_')

def sigmoid(weights, x):
    """
    Sigmoid function as presented in the CalTech lectures
    I make it overly verbose so I can see what's going on inside
    :param weights: vector of value weights, including bias weight
    :param x: vector of input values, including bias term
    :return: a number between 0 and 1
    """
    sigmoid_func = lambda s: np.exp(s) / (1.0 + np.exp(s))
    T = weights.transpose().dot(x)
    theta_s = sigmoid_func(T)
    return theta_s


def print_sigmoids(weights, x_values):
    """
    For debugging/testing, look directly at the values instead of plotting
    :param weights: vector of value weights, including bias weight
    :param x: vector of input values, including bias term
    :return: void
    """

    for x in x_values:
        print x, str(sigmoid(weights, x))

    return

def plot_sigmoid(weights, x_values, x_label, iterations):
    """
    Plots probability vs. our random variable
    :param weights: Vector of weights including bias weight
    :param x_values: Array of X values, including bias term
    :param x_label: Name of the data column we are learning
    :return:
    """
    y_label = 'Probability of True'

    y_values = []
    for x in x_values:
        y_values.append(sigmoid(weights, x))

    x_space = x_values[:, 0]
    y_space = np.array(y_values)

    mpl.plot(x_space, y_space, 'ro')
    mpl.title("Sigmoid Function for Probability of Default Given " + x_label + " After " + str(iterations) + " Iterations")
    mpl.xlabel(x_label)
    mpl.ylabel(y_label)
    mpl.autoscale(tight=True)
    mpl.grid()
    mpl.show()
    return

def plot_data_set(x_values, y_values, x_label):
    """
    :param x_values: array of x values, including bias term (for simplicity, I remove it in here
    :param y_values: array of target values
    :param x_label: name of the data we are considering
    :return:
    """

    y_label = 'Defaulted (yes/no)'

    x_space = x_values[:, 0]
    y_space = np.array(y_values)

    mpl.plot(x_space, y_space, 'go')
    mpl.title("Data for " + x_label + " Compared to Defaults")
    mpl.xlabel(x_label)
    mpl.ylabel(y_label)
    mpl.autoscale(tight=True)
    mpl.grid()
    mpl.show()
    return


#method:
#---Gradient descent
def gradient_descent(little_epsilon, eta, y_set, x_set, iterations):
    """
    :param little_epsilon: acceptable error rate
    :param eta: learning step size
    :param y_set: set of target values
    :param x_set: set of input values, including bias term (1.0)
    :return:
    """
    assert type(y_set) is np.ndarray, "y_set is not numpy array"
    assert type(x_set) is np.ndarray, "x_set is not numpy array"

    # t = number of iterations
    t = 0
    # initialize w(0) to 0
    weights_t = np.array([0.0, 0.0])

    # depending on the data set, we could continue training until
    # the gradient is smaller than our acceptable epsilon
    # otherwise, not using the little_epsilon parameter at the moment
    #while abs(grEin.any()) >= little_epsilon:

    # iterate to the next step until it is time to stop
    for i in range(0, iterations):
        # grEin = gradient of in-sample Error
        grEin = get_gradient(y_set, weights_t, x_set)

        # update the weights
        # delta w = eta * grEin
        delta = (1.0 * eta) * grEin

        # w(t + 1) = w(t) - delta(w(t))
        weights_t = weights_t - delta

        # The following block is for debugging/testing purposes
        # print delta
        #print abs(grEin)
        #print grEin
        #print eta * grEin
        #print weights_t

        # update iteration count
        t += 1

    # return the final weights vector W (weights_t)
    return weights_t


#method:
#---Get gradient at this t with w(t)
def get_gradient(y_set, weights_t, data_set):
    """
    :param y_set: the array of target values
    :param weights_t: array of weights with bias term
    :param data_set: array of training values with bias term
    :return:
    """

    assert type(y_set) is np.ndarray, "y_set is not a numpy array"
    assert type(weights_t) is np.ndarray, "weights_t is not a numpy array"
    assert type(data_set) is np.ndarray, "dataset is not a numpy matrix"

    # number of training examples
    N = len(data_set)
    # get the sum of error in the whole set
    summation_error = 0.0
    for n in range(0, N):
        Xn = np.array(data_set[n, :])
        summation_error += get_partial_gradient_sum(y_set[n], weights_t, Xn)

    # average out the error
    # gradient of in-sample error = -1/N * (summation of n in N -> (yn*Xn) / (1 + euler's constant ^ (yn * W(t).T * Xn))
    gradient = (-1.0/N) * summation_error
    return gradient


# method:
# returns the value for n : yn * Xn / (1 + e^(yn * W(t).T * Xn)
def get_partial_gradient_sum(yn, Wt, Xn):
    '''
    This is the inner term of the summation from the CalTech lectures/text
    I separate it out for readability/inspection/debugging
    It should really be in a matrix operation, this is really slow
    :param yn: target value for data row in question
    :param Wt: vector of Weights for this iteration
    :param Xn: vector of inputs for data row
    :return: Computed value for the summation term  yn * Xn / (1 + e^(yn * W(t).T * Xn)
    '''
    #assert yn is np.ndarray, "y is not a numpy array"
    #assert Xn is np.ndarray, "Xn is not a numpy array"
    #assert Wt is np.ndarray, "Wt is not a numpy array"
    #print Wt
    #print yn

    # y[n]X[n]
    top_term = yn * Xn
    # y[n] * W[t]T * X[n]
    exponent_term = yn * (Wt.transpose().dot(Xn))
    # 1 + e^exponent
    divisor = 1.0 + np.exp(exponent_term)
    partial = top_term / divisor

    return partial


#method:
#---Execute the learning and show results
def learn_class(data_set_file_location, data_column, column_name, target_column, little_epsilon, eta, iterations):
    """
    :param data_set_file_location: Location of data file to load
    :param data_column: Integer for input data column in the data file
    :param column_name: Name of the data we are considering (just a string we make up)
    :param target_column: Integer for target data column in the data file
    :param little_epsilon: Tolerable in-sample error (not currently being used)
    :param eta: Learning step size
    :return:
    """

    global dataset, classes
    # Load rows from credit_data.csv
    # Row 1 = Target, Row 3 = Balance, Row 4 = Income
    dataset = load_data_set(data_set_file_location, ",", target_column, data_column)

    # get possible classes (should be -1 and 1)
    # in case we want to check the data
    # classes = np.unique(dataset[:, 0])
    # for d in dataset:
    #     if d[0] == 1:
    #         print d

    X = np.array(dataset[:, 1:])
    if data_column == 4:
        X = np.array([x/1000.0 for x in X])
    Y_set = np.array(dataset[:, 0])

    plot_data_set(X, Y_set, column_name)

    #learn the functions
    weights = gradient_descent(little_epsilon, eta, Y_set, X, iterations)

    #return weights, intercept
    print "Coefficient: ", weights[0]
    print "Intercept: ", weights[1]

    #print_sigmoids(weights, sgd_classifier.intercept_, XT)
    plot_sigmoid(weights, X, column_name, iterations)
    return

'''
!!!!!!!!!!!!!!!!!!
!!! START HERE !!!
!!!!!!!!!!!!!!!!!!
'''
# little_epsilon: tolerable in-sample error
# (not currently being used, the error never seems to reach low enough)
little_epsilon = 0.1
# eta: learning rate
eta = 0.001
# learning iterations
iterations = 200

# target data column: 1
# input data column: 3
#learn: P(Default|Balance)
learn_class("/home/graham/Desktop/MachineLearning/Mod6/Credit_Data.csv", 3, 'Balance', 1, little_epsilon, eta, iterations)

# target data column: 1
# input data column: 4
#learn: P(Default|Income)
learn_class("/home/graham/Desktop/MachineLearning/Mod6/Credit_Data.csv", 4, 'Income (* 1000)', 1, little_epsilon, eta, iterations)

#demonstrate correct classification: P(D|B)
#P(Default|Balance)
#P(Default|Income)

