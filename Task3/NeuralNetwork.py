# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 23:08:45 2021

@author: priya
"""

import numpy as np 
from helper_functions import *
import math 

class NeuralNetwork:
    def __init__(self, num_features,  num_classes , num_neurons , activation_fxn , learning_rate, epochs , batch_size = -1):
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
        
        self.batch_size = batch_size

        self.initializeLayers()
        self.printNetworkSetup()
        
        
    def initializeLayers(self):
        for i in range(0, self.num_layers):
            self.layers.append(Layer(self.num_neurons[i], self.num_neurons[i+1], self.activation_fxn[i]))
                
        
    def printNetworkSetup(self):
        print("##############################################")
        print("Number of features:", self.num_features,
              "\nNumber of classes:", self.num_classes,
              "\nNumber of layers:", self.num_layers,
              "\nLearning rate", self.learning_rate)
        i=1
        for layer in self.layers: 
            print("Layer {} Number of Neurons: {}, Activation Function: {}".format(i, layer.num_neurons, layer.activation_fxn))
            print("Weight: {}, Bias: {}".format(layer.w.shape, layer.b.shape))
            i += 1
        print("##############################################")
        
    #forwards pass for network 
    def forwardPass(self, X):
        layer_input = X.T
        
        for i in range(0,len(self.layers)):
            layer_output = self.layers[i].forward(layer_input)
            layer_input = layer_output
            

                  
        return layer_output
    
    #backward pass for network  
    def backwardPass(self, X, y):
        
        #set up for the output layer 
        output_layer = self.layers[-1]

        output_layer.dZ = softmaxCrossEntropyPrime(output_layer.A, y)


        output_layer.dw = np.dot(output_layer.dZ, self.layers[-2].A.T)*(1/self.num_samples)
        output_layer.db = np.sum(output_layer.dZ, axis = 1, keepdims = True)*(1/self.num_samples)
        
        dA = np.dot(output_layer.w.T, output_layer.dZ)
        
        #second to last layer to first layer
        for i in range(self.num_layers-2, -1, -1):

            if i == 0: 
                dA_out = self.layers[i].backward(dA, X.T, self.num_samples)
            else:
                dA_out = self.layers[i].backward(dA, self.layers[i-1].A, self.num_samples)
                
            layer = self.layers[i]


            dA = dA_out
        
        
        
        
    def train(self, X, y):
        
        #if training in batches 
        if self.batch_size == -1:  
            self.num_samples = X.shape[0]
        else:
            self.num_samples = self.batch_size
            x_b, y_b = makeBatches(X, y, self.batch_size)
            
            

        for i in range(0,self.epochs):
            #print("epoch: ", i)
    
            if self.batch_size == -1:
                # forward pass 
                y_pred = self.forwardPass(X)
                

                #calculate loss
                loss = CrossEntropyLoss(y, y_pred.T)
    
                #backward pass 
                self.backwardPass(X, y)
            
                #update weights 
                for j in range(0, self.num_layers):
                    self.layers[j].w = self.layers[j].w - self.learning_rate*self.layers[j].dw
                    self.layers[j].b = self.layers[j].b - self.learning_rate*self.layers[j].db
                    
                accuracy =  self.predict(X, y)
                print('epoch ', i,  'loss: ', loss, 'accuracy: ', accuracy)
                    
            else:
                ###update weights for every batch 
                for k in range(0, len(x_b)):
                    # forward pass 
                    y_pred = self.forwardPass(x_b[k])
                    
                    
                    #calculate loss
                    loss = CrossEntropyLoss(y_b[k], y_pred.T)
        
                    #backward pass 
                    self.backwardPass(x_b[k], y_b[k])
                
                    #update weights 
                    for j in range(0, self.num_layers):
                        self.layers[j].w = self.layers[j].w - self.learning_rate*self.layers[j].dw
                        self.layers[j].b = self.layers[j].b - self.learning_rate*self.layers[j].db
                        
                    if k % 100 == 0:  
                        accuracy =  self.predict(x_b[k], y_b[k])
                        print('epoch ', i, 'batch ', k, 'loss: ', loss, 'accuracy: ', accuracy)
                       
        accuracy =  self.predict(X, y)
        print("Train accuracy: ",  accuracy, "%")
            

            
    #predicts output and returns accuracy 
    def predict(self, X, Y):
        y_pred = np.argmax(self.forwardPass(X), axis =0 )
        y_actual = np.argmax(Y, axis = 1 )
        #print(y_pred[0:5])
        #print(y_actual[0:5])
        accuracy = (y_pred==y_actual).mean()
        return round(accuracy*100,2) 
    
    #predicts output and displays actual vs predicted 
    def predictwithOutput(self, X, Y):
        y_pred = np.argmax(self.forwardPass(X), axis =0 )
        y_actual = np.argmax(Y, axis = 1 )
        print("total: ", y_pred.shape[0])
        correct = (y_pred==y_actual).sum()
        print(y_pred[0:30])
        print(y_actual[0:30])
        print("correct:", correct)
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

    #forward pass for a single layer     
    def forward(self, X):
        self.z = np.dot(self.w, X) + self.b
        
        if self.activation_fxn == 'ReLU':
            self.A = reLU(self.z)
        elif self.activation_fxn == 'sigmoid':
            self.A = sigmoid(self.z)
        elif self.activation_fxn == 'softmax':
            self.A = softmax(self.z)
            
        return self.A
            
    #backward pass for a single layer 
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
        
    

        
        

        
    
        
  
        
    

