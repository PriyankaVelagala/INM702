# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 22:51:48 2021

@author: priyanka
"""

from NeuralNetwork import *
from helper_functions import *

import tensorflow as tf
from matplotlib import pyplot


import numpy as np 
import math 

def main():
    

    #get data 
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()    

    #normalize features 
    X_train, X_test = normalizeX(X_train, X_test)    
    #OHE target 
    y_train, y_test = oheY(y_train, y_test)
    
    print("X train: ", X_train.shape, "Y train: ",  y_train.shape,  "X test: ", X_test.shape,  "Y_test: ", y_test.shape)
    
    myNN = NeuralNetwork()
    myNN.train(X_train, y_train)
    
    
    #t = [[ 0.41287266, -0.73082379,  0.78215209],
    #   [ 0.76983443,  0.46052273,  0.4283139 ],
    #   [-0.18905708,  0.57197116,  0.53226954]]
    #t = np.array(t)
    #print(d_reLU(t))



if __name__ == "__main__":
    main()