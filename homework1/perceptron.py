#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 00:34:56 2017

@author: canalli
"""

import numpy as np

#%%

def generate_random_dataset(N, M):
    X = np.random.uniform(low=-1, high=1, size=(M,N))
    return np.vstack([np.ones((1,N)), X])
    
def generate_random_weights(M):
    return np.random.uniform(low=-1, high=1, size=(M+1,1))

#%%

def sign(s):
    return s/np.abs(s)
    
def perceptron(w, x):
    return sign(np.dot(w.T,x))
    
def get_misclassified(X, Y, y_pred):
    for i in range(Y.shape[1]):
        if Y[0,i] != y_pred[0,i]:
            return X[:,i].reshape((X.shape[0],1)), Y[0,i]

    return None

def generate_function(w):
    def h(x):
        return perceptron(w, x)
        
    return h
    
def error_rate(X, f, g):
    Y_true = f(X)
    Y_pred = g(X)
    
    error_count = np.sum(Y_true != Y_pred)
    return error_count/Y_true.shape[1]

#%%

def perceptron_learning_algorithm(X, Y, N, M):
    w = generate_random_weights(M)
    t = 1
    while True:
        y_pred = perceptron(w, X)
        misclassified = get_misclassified(X, Y, y_pred)
        
        if misclassified is None:
            break
        
        x, y = misclassified
        w = w + y*x
        
        t += 1
    
    g = generate_function(w)
    return t, g
    
#%%

N = 100
M = 2
runs = 1000

X = generate_random_dataset(N,M)
f = generate_function(generate_random_weights(M))
Y = f(X)

#%%
iterations = []
for r in range(runs):
    t, g = perceptron_learning_algorithm(X, Y, N, M)
    iterations.append(t)
    
iterations = np.array(iterations)
print('mean elapsed time:', np.mean(iterations))

#%%

test_size = 100000

probabilities = []
for r in range(runs):
    t, g = perceptron_learning_algorithm(X, Y, N, M)
    X_test = generate_random_dataset(test_size, M)
    P_f_diff_g = error_rate(X_test, f, g)
    probabilities.append(P_f_diff_g)


probabilities = np.array(probabilities)
print("P[f(x) != g(x)] = ", np.mean(probabilities))