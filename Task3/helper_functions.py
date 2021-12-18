# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 23:39:22 2021

@author: priya
"""

#https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795
#https://github.com/Sentdex/NNfSiX/blob/master/Python/p006-Softmax-Activation.py


import math
import numpy as np 

"""Data Preprocessing"""

def normalizeX(X_train, X_test):
    
    #normalize 
    X_train = X_train/255
    X_test = X_test/255 
    
    #flatten to each sample from 28x28 to  1-d array 
    X_train = X_train.transpose()
    X_train = X_train.reshape(X_train.shape[2], (X_train.shape[1]*X_train.shape[0]))
    
    X_test = X_test.transpose()
    X_test = X_test.reshape(X_test.shape[2], (X_test.shape[1]*X_test.shape[0]))

    
    return X_train, X_test 

def oheY(Y_train, Y_test): 
    
    num_classes = Y_train.max()+1
    
    ohe_y_train = np.zeros((Y_train.size, num_classes))
    ohe_y_train[np.arange(Y_train.size), Y_train] = 1
    
    
    ohe_y_test = np.zeros((Y_test.size, num_classes))
    ohe_y_test[np.arange(Y_test.size), Y_test] = 1
    
    
    return ohe_y_train, ohe_y_test
    

"""Activation Functions and their derivatives"""

def sigmoid(x):
    return 1 / (1 + np.e ** -x)

def sigmoidPrime(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

############################

def reLU(x):
    return np.maximum(0, x)

def reLUPrime(x):
    return np.greater(x, 0).astype(int)
############################

def softmax(x):
    #exponentiation - to remove any -'ve values and rescale using max value 
    expX = np.exp(x - np.max(x))
    #normalize to get a probability distribution 
    normalized =  expX / expX.sum(axis=0, keepdims=True)
    #print('softmax output', normalized)
    return normalized 


#loss function for Categorical Cross Entropy 
def CrossEntropyLoss(y_actual, y_pred):
    #clip loss so log doesnt blow up 
    y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
    #calculate loss 
    loss = (-1)*y_actual* np.log(y_pred_clipped)
    return np.mean(loss)
    
# derivative for output layer with cross entropy + softmax activation 
def softmaxCrossEntropyPrime(y_actual, y_pred ):
    return y_pred - y_actual.T

