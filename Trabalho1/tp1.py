# -*- coding: utf-8 -*-

import numpy as np
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale

D = np.loadtxt("Banco de Dados - Infarto - Dentro-da-Amostra.txt")

X = D[:,0:3]
Y = D[:,3:4].reshape(D.shape[0],).astype(int)

# fix folds for every model, for a fair selection
kf = KFold(n_splits=10, shuffle=True, random_state=77)

models = []

for k in range(1,11):
    m = KNeighborsClassifier(n_neighbors=k)
    models.append( (m, "knn_%d" % k) )
    
clf = GaussianNB()
models.append( (clf, "naive_bayes") ) 

sgd = SGDClassifier(loss="perceptron", eta0=1, learning_rate="constant", penalty=None)
models.append( (sgd, "perceptron") )

sgd = SGDClassifier(loss="perceptron", eta0=1, learning_rate="constant", penalty="l1")
models.append( (sgd, "perceptron+l1") ) 

sgd = SGDClassifier(loss="perceptron", eta0=1, learning_rate="constant", penalty="l2")
models.append( (sgd, "perceptron+l2") ) 

sgd = SGDClassifier(loss="hinge", penalty=None)
models.append( (sgd, "linear_svm") )

sgd = SGDClassifier(loss="hinge", penalty="l1")
models.append( (sgd, "linear_svm+l1") ) 

sgd = SGDClassifier(loss="hinge", penalty="l2")
models.append( (sgd, "linear_svm+l2") ) 

sgd = SGDClassifier(loss="log", penalty=None)
models.append( (sgd, "log_regression") )

sgd = SGDClassifier(loss="log", penalty="l1")
models.append( (sgd, "log_regression+l1") ) 

sgd = SGDClassifier(loss="log", penalty="l2")
models.append( (sgd, "log_regression+l2") ) 

for c in np.arange(0.5, 2, 0.1):

    clf = SVC(C=c, kernel="rbf")
    models.append( (clf, "rbf_svm_C%.1f" % c) )

for (model, name) in models:    
    scores = [] 
    for train_index, test_index in kf.split(X):
        model.fit(scale(X[train_index]), Y[train_index])
        score = model.score(scale(X[test_index]), Y[test_index])
        scores.append(score)
    
    #print("--------", name, "--------")
    #print("precision\t\t", np.mean(scores))
    #print("standard deviation\t", np.std(scores))
    #print("\n")
    print(name, "& $", "%.4f" % np.mean(scores), "$ & $", "%.4f" % np.std(scores), "$\\\\")


