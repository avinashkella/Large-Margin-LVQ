"""
Created on Thu Sep 17 12:43:51 2020
@author: avinash
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import *

class LMLVQ:
    """ Large margin LVQ is to maximize the distance of sample margin or
    to maximize the distance between decision hyperplane and data point.

    Attributes:
        prototype_per_class:
            The number of prototypes per class to be learned.
    """
    def __init__(self, prototype_per_class):
        """Inits LMLVQ with prototypes per class."""
        self.prototype_per_class = prototype_per_class

    update_prototypes = np.array([])
    prt_labels = np.array([])


    def normalization(self, input_value):
        """ Normalize the data between range 0 and 1.

        Args:
            input_value:
                A n x m matrix of input data.

        Return:
            normalized_data:    A n x m matrix with values between 0 and 1.
        """
        minimum = np.amin(input_value, axis=0)
        maximum = np.amax(input_value, axis=0)
        normalized_data = (input_value - minimum)/(maximum - minimum)
        return normalized_data


    def prt(self, input_data, data_labels, prototype_per_class):
        """ Calculate prototypes with labels either at mean or randomly depends
        on prototypes per class.

        Args:
            input_value:
                A n x m matrix of datapoints.
            data_labels:
                A n-dimensional vector containing the labels for each
                datapoint.
            prototypes per class:
                The number of prototypes per class to be learned. If it is
                equal to 1 then prototypes assigned at mean position
                else it assigns randolmy.

        Return:
            prototype_labels:
                A n-dimensional vector containing the labels for each
                prototype.
            prototypes:
                A n x m matrix of prototyes.for training.
        """
        #calculate prototype_labels
        prototype_labels = np.unique(data_labels)
        prototype_labels = list(prototype_labels) * prototype_per_class
        #calculate prototypes
        prt_labels = np.expand_dims(prototype_labels, axis=1)
        expand_dimension = np.expand_dims(np.equal(prt_labels, data_labels),
                                          axis=2)
        count = np.count_nonzero(expand_dimension, axis=1)
        proto = np.where(expand_dimension, input_data, 0)
        #if prototype_per_class is 1 then assign it to mean else assign prototypes randomly
        prototypes_array = []
        if prototype_per_class == 1:
            prototypes = np.sum(proto, axis=1)/count
        else:
            for lbl in range(len(prototype_labels)):
                x_values = input_data[data_labels == prototype_labels[lbl]]
                num = np.random.choice(x_values.shape[0], 1, replace=False)
                prototypes = x_values[num, :]
                prototypes_array.append(prototypes)
            prototypes = np.squeeze(np.array(prototypes_array))
        #save prototype labels
        self.prt_labels = prototype_labels
        return self.prt_labels, prototypes

    def euclidean_dist(self, input_data, prototypes):
        """ Calculate squared Euclidean distance between datapoints and
        prototypes.

        Args:
            input_data:
                A n x m matrix of datapoints.
            prototpes:
                A n x m matrix of prototyes of each class.

        Return:
            A n x m matrix with Euclidean distance between datapoints and
            prototypes.
        """
        expand_dimension = np.expand_dims(input_data, axis=1)
        distance = expand_dimension - prototypes
        distance_square = np.square(distance)
        sum_distance = np.sum(distance_square, axis=2)
        eu_dist = np.sqrt(sum_distance)
        return eu_dist

    def distance_plus(self, data_labels, prototype_labels,
                      prototypes, eu_dist):
        """ Calculate squared Euclidean distance between datapoints and
        prototypes with same labels.

        Args:
            data_labels:
                A n-dimensional vector containing the labels for each
                datapoint.
            prototype_labels:
                A n-dimensional vector containing the labels for each
                prototype.
            prototpes:
                A n x m matrix of prototyes of each class.
            eu_dist:
                A n x m matrix with Euclidean distance between datapoints
                and prototypes.

        Return:
            d_plus:
                A n-dimensional vector containing distance between
                datapoints and prototypes with same label.
            w_plus:
                A m x n matrix of nearest correct matching prototypes.
            w_plus_index:
                A n-dimensional vector containing the indices for
                nearest prototypes to datapoints with same label.
        """
        expand_dimension = np.expand_dims(prototype_labels, axis=1)
        label_transpose = np.transpose(np.equal(expand_dimension, data_labels))

        # distance of matching prototypes
        plus_dist = np.where(label_transpose, eu_dist, np.inf)
        d_plus = np.min(plus_dist, axis=1)

        # index of minimum distance for best matching prototypes
        w_plus_index = np.argmin(plus_dist, axis=1)
        w_plus = prototypes[w_plus_index]
        return d_plus, w_plus, w_plus_index

    def d_il(self, data_labels, prototype_labels, prototypes, eu_dist):
        """ Calculate squared Euclidean distance between datapoints and
        prototypes with different labels.

        Args:
            data_labels:
                A n-dimensional vector containing the labels for each
                datapoint.
            prototype_labels:
                A n-dimensional vector containing the labels for each
                prototype.
            prototpes:
                A n x m matrix of prototyes of each class.
            eu_dist:
                A n x m matrix with Euclidean distance between datapoints
                and prototypes.

        Return:
            A n x m matrix with Euclidean distance between datapoints and
                prototypes of different labels.
        """
        expand_dime = np.expand_dims(prototype_labels, axis=1)
        #calculate data labal and protytype label are not same
        label_transpose = np.transpose(np.not_equal(expand_dime, data_labels))
        in_dist = np.where(label_transpose, eu_dist, np.nan)
        return in_dist

    def relu(self, x_value):
        """ An activation function.

        Args:
            x_value:
                A n x m matrix of datapoints or any value.

        Return:
            it return the value if it is greater than 0 otherwise returns 0.
        """
        return np.maximum(0, x_value)

    def cost_function(self, dplus, in_dist, constant, margin):
        """ Calculate cost function of LMLVQ.
        cost function: \sum_{i=1}^{m} d_{i}^+ + \frac{1}{2C} \cdot
                        \sum_{i=1}^{m} \sum_{l \in I_i}
                        ReLU(d_{i}^+ - d_{i,l} + \gamma)^2.

        Args:
            d_plus:
                A n-dimensional vector containing distance between
                datapoints and prototypes with same label.
            in_dist:
                A n x m matrix with Euclidean distance between datapoints
                and prototypes of different labels.
            Constant:
                The regularization constant.
            margin:
                The margin parameter.

        Return:
            The value which is the sum of all local errors.
        """
        expand_dimension = np.expand_dims(dplus, axis=1)
        push_ones = np.ones(in_dist.shape)
        same_dimension = expand_dimension * push_ones
        #push in cost function
        formula_push = np.square((same_dimension - in_dist + margin))
        push_relu = self.relu(formula_push)
        #replace nan with 0
        nan_in_array = isnan(push_relu)
        push_relu[nan_in_array] = 0
        #calculate push and sum
        push_sum = np.sum(push_relu, axis=1)
        #calculate pull and push
        pull = np.sum(expand_dimension, axis=0)
        push = np.dot((1/(2*constant)), np.sum(push_sum, axis=0))
        #calculate cost function
        result = pull + push
        return result

    def w_update(self, input_data, data_labels, prototypes, w_plus_index,
                 dplus, in_dist, constant, margin, _lr):
        """ Calculate the update of prototypes.
        update function: w(t+1) = w(t) - \eta * \frac{\partial cost_function}
                                                    {\partial w(t)}.

        Args:
            input_data:
                A n x m matrix of datapoints.
            data_labels:
                A n-dimensional vector containing the labels for each
                datapoint.
            prototpes:
                A n x m matrix of prototyes of each class.
            w_plus_index:
                A n-dimensional vector containing the indices for
                nearest prototypes to datapoints with same label.
            d_plus:
                A n-dimensional vector containing distance between
                datapoints and prototypes with same label.
            in_dist:
                A n x m matrix with Euclidean distance between datapoints
                and prototypes of different labels.
            Constant:
                The regularization constant.
            margin:
                The margin parameter.
            _lr:
                Learning rate also called step size.

        Return:
            The result of updated prototypes after calculated the update of
            prototypes with same and different labels.
        """
        #calculate pull for correct prototypes
        pull = []
        for k in range(len(self.prt_labels)):
            pull_values = -2 * np.sum(input_data[w_plus_index == k, :]-prototypes[k, :], 0)
            pull.append(pull_values)
        pull = np.array(pull)

        #calculate push for correct prototypes
        push_correct = []

        for k in range(len(prototypes)):
            for lbl in range(len(np.unique(data_labels))):
                #because l is prototypes with different labels
                x_values = self.prt_labels[k]
                y_values = self.prt_labels[lbl]
                if x_values != y_values:
                    array_values = []
                    #calculate d^+ - d_{i,l} + gamma
                    res = dplus[w_plus_index == k]-in_dist[w_plus_index == k, lbl]+margin
                    #calculte x_i - w_k
                    res2 = input_data[w_plus_index == k, :]-prototypes[k, :]
                    #product of d^+ - d_{i,l} + gamma with x_i - w_k
                    total = np.dot(res, res2)
                    array_values.append(total)
            d_array = np.array(array_values)
            sum_l = np.sum(d_array, axis=0)
            push_correct.append(2/constant * sum_l)
        push_correct = np.array(push_correct)
        #calculate push for incorrect prototypes
        push_incorrect = []
        nan_in_array = isnan(in_dist)
        in_dist[nan_in_array] = 0
        for k in range(len(prototypes)):
            #calculate d^+ - d_{i,k} + gamma
            res = dplus[w_plus_index != k]-in_dist[w_plus_index != k, k]+margin
            #calculte x_i - w_k
            res2 = input_data[w_plus_index != k, :]-prototypes[k, :]
            #product of d^+ - d_{i,l} + gamma with x_i - w_k
            total = 2/constant * np.dot(res, res2)
            push_incorrect.append(total)
        push_incorrect = np.array(push_incorrect)

        cost_derivative = pull - push_correct + push_incorrect

        #update weight
        proto = prototypes - (_lr * cost_derivative)
        return proto

    def plot(self, input_data, data_labels, prototypes, prototype_labels):
        """ Scatter plot of data and prototypes.

        Args:
            input_data:
                A n x m matrix of datapoints.
            data_labels:
                A n-dimensional vector containing the labels for each
                datapoint.
            prototpes:
                A n x m matrix of prototyes of each class.
            prototype_labels:
                A n-dimensional vector containing the labels for
                each prototype.
        """
        plt.scatter(input_data[:, 0], input_data[:, 2], c=data_labels,
                    cmap='viridis')
        plt.scatter(prototypes[:, 0], prototypes[:, 2], c=prototype_labels,
                    s=60, marker='D', edgecolor='k')

    def fit(self, input_data, data_labels, learning_rate, epochs, margin, constant):

        Args:
            input_data:
                A m x n matrix of  distances. Note that we have no
                preconditions for this matrix.
            data_labels:
                A m dimensional label vector for the data points.
            learning_rate:
                The step size.
            epochs:
                The maximum number of optimization iterations.
            margin:
                The margin parameter.
            Constant:
                The regularization constant.
        """
        normalized_data = self.normalization(input_data)
        prototype_l, prototypes = self.prt(normalized_data, data_labels,
                                           self.prototype_per_class)
        error = np.array([])
        plt.ion()
        plt.subplots(8, 8)
        for i in range(epochs):
            eu_dist = self.euclidean_dist(normalized_data, prototypes)

            d_plus, w_plus, w_plus_index = self.distance_plus(data_labels,
                                                              prototype_l,
                                                              prototypes,
                                                              eu_dist)
            in_dist = self.d_il(data_labels, prototype_l,
                                prototypes, eu_dist)
            cost_fun = self.cost_function(d_plus, in_dist, constant, margin)
            prototypes = self.w_update(normalized_data, data_labels,
                                       prototypes, w_plus_index,
                                       d_plus, in_dist, constant, margin,
                                       learning_rate)

            err = self.relu(cost_fun)
            change_in_error = 0

            if i == 0:
                change_in_error = 0
            else:
                change_in_error = error[-1] - err
            error = np.append(error, err)
            print("Epoch : {}, Error : {} Error change : {}".format(
                i + 1, err, change_in_error))

            plt.subplot(1, 2, 1)
            #self.plot(normalized_data, data_labels, prototypes, prototype_l)
            plt.scatter(normalized_data[:, 0], normalized_data[:, 1], c=data_labels,
                        cmap='viridis')
            plt.scatter(prototypes[:, 0], prototypes[:, 1], c=prototype_l,
                        s=60, marker='D', edgecolor='k')
            plt.subplot(1, 2, 2)
            plt.plot(np.arange(i + 1), error, marker="d")
            plt.show()
            plt.pause(0.05)
            plt.clf()
        self.update_prototypes = prototypes
        print(self.update_prototypes)
        return self.update_prototypes

    def predict(self, input_value):
        """ Predicts the labels for the data represented by the
        given test-to-training distance matrix. Each datapoint will be assigned
        to the closest prototype.

        Args:
            input_value:
                A n x m matrix of distances from the test to the training
                datapoints.

        Return:
            ylabel:
                A n-dimensional vector containing the predicted labels for each
                datapoint.
        """
        input_value = self.normalization(input_value)
        prototypes = self.update_prototypes
        eu_dist = self.euclidean_dist(input_value, prototypes)
        m_d = np.min(eu_dist, axis=1)
        expand_dims = np.expand_dims(m_d, axis=1)
        ylabel = np.where(np.equal(expand_dims, eu_dist),
                          self.prt_labels, np.inf)
        ylabel = np.min(ylabel, axis=1)
        print(ylabel)
        return ylabel
