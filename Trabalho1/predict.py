#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 10:16:02 2017

@author: canalli
"""

import sys
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import scale

D = np.loadtxt("Banco de Dados - Infarto - Dentro-da-Amostra.txt")
X = D[:,0:3]
Y = D[:,3:4].reshape(D.shape[0],).astype(int)

test_path = sys.argv[1]
X_test = np.loadtxt(test_path)[:,0:3]

clf = GaussianNB()
clf.fit(scale(X), Y)

y_pred = clf.predict(scale(X_test))

for y in y_pred:
    print(y)
    
np.savetxt("predict.csv", np.array(y_pred, dtype=int), fmt="%d")



