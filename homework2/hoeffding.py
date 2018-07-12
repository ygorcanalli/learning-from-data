#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 03:25:25 2017

@author: canalli
"""

import numpy as np

runs = 100000

v_1_array = []
v_rand_array = []
v_min_array = []

for r in range(runs):
    coins = np.random.randint(0, 2, size=(1000,10))
    
    c_1 = 0
    c_rand = np.random.randint(0, 1000)
    
    head_count = np.sum(coins, axis=1)
    c_min = int(np.argmin(head_count))
    
    v_1 = head_count[c_1] / 10
    v_rand = head_count[c_rand] / 10
    v_min = head_count[c_min] / 10

    v_1_array.append(v_1)
    v_rand_array.append(v_rand)
    v_min_array.append(v_min)
    
v_1_array = np.array(v_1_array)
v_rand_array = np.array(v_rand_array)
v_min_array = np.array(v_min_array)

print("v_1 =", np.mean(v_1_array))
print("v_rand =", np.mean(v_rand_array))
print("v_min =", np.mean(v_min_array))