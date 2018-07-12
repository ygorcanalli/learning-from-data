#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 11:30:49 2017

@author: canalli
"""

# -*- coding: utf-8 -*-

import numpy as np
from pprint import pprint
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
from keras.metrics import binary_accuracy
from sklearn.metrics import confusion_matrix, recall_score
   
def create_model():
    es = EarlyStopping(monitor='val_loss', patience=3)

    model = Sequential([
        Dropout(0.2, input_shape=(3,)),
        Dense(1000),
        Activation('relu'),
        Dropout(0.2),
        Dense(1000),
        Activation('relu'),
        Dropout(0.2),
        Dense(2),
        Activation('softmax'),
    ])

    #sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    fitting_args = {
        'epochs': 1000,
        'batch_size': 6,
        'verbose': 1,
        'validation_split': 1/7,
        'callbacks': [es]
    }
    
    return model, fitting_args
    
def load_dataset(path):
    D = np.loadtxt(path)

    X = D[:,0:3]
    y = D[:,3:4].reshape(D.shape[0],)
    
    return X, y
    
def encode_target(y):
    y_encoded = np.zeros( (y.shape[0], 2) )
    
    y_neg_index = y == -1
    y_pos_index = y == 1
    
    y_encoded[y_neg_index,0] = 1
    y_encoded[y_pos_index,1] = 1

    return y_encoded
    
def decode_target(y):
    y_decoded = np.zeros( (y.shape[0],) )
    
    y_neg_index = y == 0
    y_pos_index = y == 1

    y_decoded[y_neg_index] = -1
    y_decoded[y_pos_index] = 1

    return y_decoded
    
def accuracy_score(y_true, y_pred):
    return sum(y_true == y_pred)/y_true.shape[0]
    
def evaluate_model(X, y):
    # fix folds for every model, for a fair selection
    kf = KFold(n_splits=5, shuffle=True)
    
    Y = encode_target(y)
    
    scores = [] 
    for train_index, test_index in kf.split(X):
        model, args = create_model()
        model.fit(scale(X[train_index]), Y[train_index], **args)
        score = model.evaluate(scale(X[test_index]), Y[test_index], batch_size=args['batch_size'])
        scores.append(score[1])
    
    evaluation = {
        'metric': 'accuracy',
        'scores': scores,
        'mean': np.mean(scores),
        'std': np.std(scores)
    }        
    
    return evaluation
    
def fit(X, y):
    
    Y = encode_target(y)
    model, args = create_model()
    model.fit(scale(X), Y, **args)
    
    return model

def predict(X, model):
    
    Y_pred = model.predict_classes(scale(X))
    y_pred = decode_target(Y_pred)
    
    return y_pred
    


#%%
X, y = load_dataset("Banco de Dados - Infarto - Dentro-da-Amostra.txt")
#X, y = load_dataset("train.csv")
#X_true, y_true = load_dataset("test.csv")

#%%
evaluation = evaluate_model(X, y)
pprint(evaluation)
#%%

model = fit(X, y)
y_pred = predict(X_true, model)
print('\nacc: ', accuracy_score(y_true, y_pred))
print('recall: ',recall_score(y_true, y_pred))
print('confusion: ', confusion_matrix(y_true, y_pred))
#%%



