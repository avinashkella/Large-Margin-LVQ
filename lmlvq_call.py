#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 16:27:52 2020

@author: avinash
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from lmlvq_numpy import LMLVQ
import numpy as np

prototype_per_class = 1
input_data = load_iris().data
data_label = load_iris().target
epochs = 1
learning_rate = 0.01
margin = 1
C = 1E2



clf = LMLVQ(prototype_per_class)

X_train, X_test, y_train, y_test = train_test_split(input_data,
                                                    data_label,
                                                    test_size=0.3,
                                                    random_state=42)

clf.fit(X_train, y_train, learning_rate, epochs, margin, C)

y_predict = clf.predict(X_test)

acc = np.sum(y_predict == y_test)
acc = acc / len(y_test) * 100
print(acc)





