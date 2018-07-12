#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 00:34:27 2017

@author: canalli
"""

import numpy as np
from .perceptron import generate_random_dataset, generate_function, generate_random_weights

def linear_regression(X):
    return None

N = 100
M = 2
runs = 1000

X = generate_random_dataset(N,M)
f = generate_function(generate_random_weights(M))
Y = f(X)
