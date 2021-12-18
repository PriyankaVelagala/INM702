# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 23:08:45 2021

@author: priya
"""

import numpy as np 
from helper_functions import *
import math 

class NeuralNetwork:
    
    def __init__(self, num_features = 784,  num_classes = 10,  num_neurons = [50,30],
                 activation_fxn = ['sigmoid', 'ReLU'],  learning_rate = 0.01, epochs = 2):
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_neurons = num_neurons
        
        # add for input_layer 
        self.num_neurons.insert(0, num_features)

        self.activation_fxn = activation_fxn 
        self.learning_rate = learning_rate 
        
        #add for output layer 
        self.num_neurons.append(num_classes)
        self.activation_fxn.append('softmax')
    
        
        
        self.num_layers = len(activation_fxn)
        
        self.layers = []
        
        self.epochs = epochs
        self.num_samples = -1 

        self.initializeLayers()
        self.printNetworkSetup()
        
        
    def initializeLayers(self):
        #first layer 
        #self.layers.append(Layer(self.num_features, self.num_neurons[0], self.activation_fxn[0])) 
        #hidden layers - 2 to n
        for i in range(0, self.num_layers):
            self.layers.append(Layer(self.num_neurons[i], self.num_neurons[i+1], self.activation_fxn[i]))
                
        
    def printNetworkSetup(self):
        print("Number of features:", self.num_features,
              "\nNumber of classes:", self.num_classes,
              "\nLearning rate", self.learning_rate)
        i=1
        for layer in self.layers: 
            print("Layer {} Number of Neurons: {}, Activation Function: {}".format(i, layer.num_neurons, layer.activation_fxn))
            print("Weight: {}, Bias: {}".format(layer.w.shape, layer.b.shape))
            i += 1
        
    #forwards pass 
    def forwardPass(self, X):
        layer_input = X.T
        
        for i in range(0,len(self.layers)):
            layer_output = self.layers[i].forward(layer_input)
            layer_input = layer_output
            
            #print("Layer ", i+1, "Activation fxn: ", layer.activation_fxn)
            #print(" Z shape: {}, A shape: {}".format(layer.z.shape, layer.A.shape))
                  
        return layer_output
    
    #backward pass 
    def backwardPass(self, X, y):
        
        #set up for the output layer 
        output_layer = self.layers[self.num_layers]
        output_layer.dZ = softmaxCrossEntropyPrime(output_layer.A, y)
        output_layer.dw = np.dot(output_layer.dZ, self.layers[self.num_layers-1].A.T)*(1/self.num_samples)
        output_layer.db = np.sum(output_layer.dZ, axis = 1, keepdims = True)*(1/self.num_samples())
        
        dA = np.dot(output_layer.w.T, output_layer.dZ)

        for i in range(self.num_layers, 0, -1):
            if i == 1: 
                dA_out = self.layers[i].backward(dA, X.T, self.num_samples)
            else:
                dA_out = self.layers[i].backward(dA, self.layers[i-1].A, self.num_samples)

                
            dA = dA_out
        
        
        
        
    def train(self, X, y):
        self.num_samples = X.shape[0]
        
        #for X iterations
        for i in range(0,self.epochs):
            # forward pass 
            y_pred = self.forwardPass(X)
            
            #print("Y_pred shape: ", y_pred.shape)
            #print("y-shape: ", y.shape)
            
            #calculate loss
            loss = CrossEntropyLoss(y, y_pred.T)
            print('loss:', loss)

            #backward pass 
            temp = backwardPass(X, y)
        
            #update weights 
            
               
        
        
class Layer:
    
    def __init__(self, num_inputs, num_neurons, activation_fxn):
        self.num_inputs = num_inputs 
        self.num_neurons = num_neurons 
        self.activation_fxn = activation_fxn
        
        #initialize weights and biases 
        self.w = 0.1*np.random.randn(num_neurons, num_inputs)
        self.b = 0.1*np.zeros((num_neurons, 1))
        
        self.z = None
        self.A = None 
       
        #for backward 
        self.dZ = None 
        self.dA = None 
        self.dw =None
        self.db = None 

        
        
    def forward(self, X):
        self.z = np.dot(self.w, X) + self.b
        
        if self.activation_fxn == 'ReLU':
            self.A = reLU(self.z)
        elif self.activation_fxn == 'sigmoid':
            self.A = sigmoid(self.z)
        elif self.activation_fxn == 'softmax':
            self.A = softmax(self.z)
            
        return self.A
            
    def backward(self, dA, X, num_samples):
        
        if self.activation_fxn == 'ReLU':
            z_prime = reLUPrime(self.z)
        elif self.activation_fxn == 'sigmoid':
            z_prime= sigmoidPrime(self.z)

        
        self.dZ = dA * z_prime 
        self.dW = np.dot(self.dZ, X.T) * (1/num_samples)
        self.db = np.sum(self.dZ, axis = 1, keepdims = True )
        self.dA = np.dot(self.W.T, self.dZ)
        
        return self.dA
        
    

        
        

        
    
        
  
        
    

