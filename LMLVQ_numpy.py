#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 12:43:51 2020

@author: avinash
"""
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy import *
from random import random
from sklearn.cluster import KMeans

class lmlvq:

    def __init__(self, prototype_per_class):
        self.prototype_per_class = prototype_per_class

    update_prototypes = np.array([])
    prt_labels = np.array([])

    # normalize the data
    def normalization(self, input_data):
        minimum = np.amin(input_data, axis=0)
        maximum = np.amax(input_data, axis=0)
        normalized_data = (input_data - minimum)/(maximum - minimum)
        return normalized_data

    # define prototypes
    def prt(self, input_data, data_labels, prototype_per_class):

        # prototype_labels are
        prototype_labels = np.unique(data_labels)
        prototype_labels = list(prototype_labels) * prototype_per_class
        
        # prototypes are
        prt_labels = np.expand_dims(prototype_labels, axis=1)
        expand_dimension = np.expand_dims(np.equal(prt_labels, data_labels),
                                          axis=2)
        
        count = np.count_nonzero(expand_dimension, axis=1)
        proto = np.where(expand_dimension, input_data, 0)
        
        p = []
        if prototype_per_class == 1:
            prototypes = np.sum(proto, axis=1)/count
        else:
            for l in range(len(prototype_labels)):
                x = input_data[data_labels == prototype_labels[l]]
                c = np.random.choice(x.shape[0], 1, replace = False)
                prototypes = x[c,:]
                p.append(prototypes)
            prototypes = np.squeeze(np.array(p))
        
        
        self.prt_labels = prototype_labels
        return self.prt_labels, prototypes

    # define euclidean distance
    def euclidean_dist(self, input_data, prototypes):
        expand_dimension = np.expand_dims(input_data, axis=1)
        distance = expand_dimension - prototypes
        distance_square = np.square(distance)
        sum_distance = np.sum(distance_square, axis=2)
        eu_dist = np.sqrt(sum_distance)
        return eu_dist

    # define d_plus
    def distance_plus(self, data_labels, prototype_labels,
                      prototypes, eu_dist):
        expand_dimension = np.expand_dims(prototype_labels, axis=1)
        label_transpose = np.transpose(np.equal(expand_dimension, data_labels))

        # distance of matching prototypes
        plus_dist = np.where(label_transpose, eu_dist, np.inf)
        d_plus = np.min(plus_dist, axis=1)

        # index of minimum distance for best matching prototypes
        w_plus_index = np.argmin(plus_dist, axis=1)
        
        w_plus = prototypes[w_plus_index]
        return d_plus, w_plus, w_plus_index

    # define d_{i,l}
    def d_il(self, data_labels, prototype_labels, prototypes, eu_dist):
        expand_dimension = np.expand_dims(prototype_labels, axis=1)
        label_transpose = np.transpose(np.not_equal(expand_dimension, data_labels))
        
        #data labal and protytype label are not same
        in_dist = np.where(label_transpose, eu_dist, np.nan)
        return in_dist
    
    #define relu function
    def relu(self, X):
        return np.maximum(0,X)

    # define cost function
    def cost_function(self, dplus, in_dist, C, gamma):
        expand_dimension = np.expand_dims(dplus, axis=1)
        push_ones = np.ones(in_dist.shape)
        same_dimension = expand_dimension * push_ones
        #push in cost function
        formula_push = np.square((same_dimension - in_dist + gamma))
        push_relu = self.relu(formula_push)
        #replace nan with 0
        where_are_NaNs = isnan(push_relu)
        push_relu[where_are_NaNs] = 0
        #calculate push and sum
        push_sum = np.sum(push_relu, axis = 1)
        #calculate pull and push
        pull = np.sum(expand_dimension, axis = 0)
        push = np.dot((1/(2*C)) , np.sum(push_sum, axis=0))
        #calculate cost function
        cf = pull + push
        return cf

    def w_update(self, input_data, data_labels, prototypes, w_plus_index, dplus, in_dist, C, gamma, lr):
        #calculate pull for correct prototypes
        pull = []
        for k in range(len(self.prt_labels)):
            p = -2 * np.sum(input_data[w_plus_index == k,:]-prototypes[k,:], 0)
            pull.append(p)
        pull = np.array(pull)
        
        #calculate push for correct prototypes
        push_correct = []
        
        for k in range(len(prototypes)):
            for l in range (len(np.unique(data_labels))):
                #because l is prototypes with different labels
                x = self.prt_labels[k]
                y = self.prt_labels[l]
                if x!=y:
                    D = []
                    #calculate d^+ - d_{i,l} + gamma
                    res = dplus[w_plus_index == k]-in_dist[w_plus_index == k, l]+gamma
                    #print(in_dist[w_plus_index == k, l])
                    #calculte x_i - w_k
                    res2 = input_data[w_plus_index == k,:]-prototypes[k,:]
                    #product of d^+ - d_{i,l} + gamma with x_i - w_k
                    total = np.dot(res, res2)
                    D.append(total)
            X = np.array(D)
            sum_l = np.sum(X,axis = 0)
            push_correct.append(2/C * sum_l)
        push_correct = np.array(push_correct)
        
        #calculate push for incorrect prototypes
        push_incorrect = []
        where_are_NaNs = isnan(in_dist)
        in_dist[where_are_NaNs] = 0
        for k in range(len(prototypes)):
            #calculate d^+ - d_{i,k} + gamma
            res = dplus[w_plus_index != k]-in_dist[w_plus_index != k, k]+gamma
            #calculte x_i - w_k
            res2 = input_data[w_plus_index != k,:]-prototypes[k,:]
            #product of d^+ - d_{i,l} + gamma with x_i - w_k
            total = 2/C * np.dot(res, res2)
            push_incorrect.append(total)
        push_incorrect = np.array(push_incorrect)
        
        cost_derivative = pull - push_correct + push_incorrect
        
        
        #update weight
        proto = prototypes - (lr * cost_derivative)
        return proto

    # plot  data
    def plot(self, input_data, data_labels, prototypes, prototype_labels):
        plt.scatter(input_data[:, 0], input_data[:, 2], c=data_labels,
                    cmap='viridis')
        plt.scatter(prototypes[:, 0], prototypes[:, 2], c=prototype_labels,
                    s=60, marker='D', edgecolor='k')

    # fit function
    def fit(self, input_data, data_labels, learning_rate, epochs, gamma, C):
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

            cf = self.cost_function(d_plus, in_dist, C, gamma)
            prototypes = self.w_update(normalized_data, data_labels, prototypes, w_plus_index, 
                                       d_plus, in_dist, C, gamma, learning_rate)
            
            
            err = self.relu(cf)
            change_in_error = 0
            if (i == 0):
                change_in_error = 0

            else:
                change_in_error = error[-1] - err
            error = np.append(error, err)
            print("Epoch : {}, Error : {} Error change : {}".format(
                i + 1, err, change_in_error))
            
            plt.subplot(1, 2, 1)
            #self.plot(normalized_data, data_labels, prototypes, prototype_l)
            plt.scatter(normalized_data[:, 0], normalized_data[:, 2], c=data_labels,
                    cmap='viridis')
            plt.scatter(prototypes[:, 0], prototypes[:, 2], c=prototype_l,
                    s=60, marker='D', edgecolor='k')
            plt.subplot(1, 2, 2)
            plt.plot(np.arange(i + 1), error, marker="d")
            plt.show()
            plt.pause(0.05)
            plt.clf()
        self.update_prototypes = prototypes
        return self.update_prototypes
    
    # data predict
    def predict(self, input_value):
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
