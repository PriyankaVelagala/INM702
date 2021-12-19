# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 23:08:45 2021

@author: priya
"""

import numpy as np 
from helper_functions import *
import math 

class NeuralNetwork:
    
    def __init__(self, num_features = 784,  num_classes = 10,  num_neurons = [300, 200, 50],
                 activation_fxn = ['sigmoid', 'sigmoid', 'ReLU'],  learning_rate = 0.1, epochs = 50):
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
              "\nNumber of layers:", self.num_layers,
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
        output_layer = self.layers[-1]
        #print("Output Layer A  shape: {}".format(output_layer.A.shape))

        output_layer.dZ = softmaxCrossEntropyPrime(output_layer.A, y)
        #print("Output Layer dZ shape: {}".format(output_layer.dZ.shape))
        #print("Output Layer -1 A  shape: {}".format(self.layers[-2].A.shape))

        output_layer.dw = np.dot(output_layer.dZ, self.layers[-2].A.T)*(1/self.num_samples)
        output_layer.db = np.sum(output_layer.dZ, axis = 1, keepdims = True)*(1/self.num_samples)
        
        #print ("Output Layer dZ shape: {}, dw shape: {}, db shape: {} ".format(output_layer.dZ.shape, output_layer.dw.shape, output_layer.db.shape))

        dA = np.dot(output_layer.w.T, output_layer.dZ)
        
        #print("dA shape: {}".format(dA.shape))
        
        #print("list range: ", list(range(self.num_layers-2, -1, -1)))

        for i in range(self.num_layers-2, -1, -1):
            #print(i)
            #layer = self.layers[i-1]
            #print("Layer {} Number of Neurons: {}, Activation Function: {}".format(i, layer.num_neurons, layer.activation_fxn))

            if i == 0: 
                dA_out = self.layers[i].backward(dA, X.T, self.num_samples)
            else:
                dA_out = self.layers[i].backward(dA, self.layers[i-1].A, self.num_samples)
                
            layer = self.layers[i]

            #print ("Layer {} dZ shape: {}, dw shape: {}, db shape: {}, dA shape: {} ".format(i+1, layer.dZ.shape, layer.dw.shape, layer.db.shape, layer.dA.shape))

            dA = dA_out
        
        
        
        
        
    def train(self, X, y):
        self.num_samples = X.shape[0]
        
        #for X iterations
        for i in range(0,self.epochs):
            print(i)
            # forward pass 
            y_pred = self.forwardPass(X)
            
            #print("Y_pred shape: ", y_pred.shape)
            #print("y-shape: ", y.shape)
            
            #calculate loss
            loss = CrossEntropyLoss(y, y_pred.T)

            #backward pass 
            self.backwardPass(X, y)
        
            #update weights 
            for j in range(0, self.num_layers):
                self.layers[j].w = self.layers[j].w - self.learning_rate*self.layers[j].dw
                self.layers[j].b = self.layers[j].b - self.learning_rate*self.layers[j].db
                
            if i % 5 == 0:  
                accuracy =  self.predict(X, y)
                print('loss:', loss, 'accuracy: ', accuracy)
            

            
    
    def predict(self, X, Y):
        y_pred = np.argmax(self.forwardPass(X), axis =0 )
        y_actual = np.argmax(Y, axis = 1 )
        #print(y_pred[0:5])
        #print(y_actual[0:5])
        accuracy = (y_pred==y_actual).mean()
        return round(accuracy*100,2) 
        

        
            
               
        
        
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
            act_prime = reLUPrime(self.z)
        elif self.activation_fxn == 'sigmoid':
            act_prime= sigmoidPrime(self.z)

        
        self.dZ = dA * act_prime 
        self.dw = np.dot(self.dZ, X.T) * (1/num_samples)
        self.db = np.sum(self.dZ, axis = 1, keepdims = True )
        self.dA = np.dot(self.w.T, self.dZ)
        
        return self.dA
        
    

        
        

        
    
        
  
        
    

