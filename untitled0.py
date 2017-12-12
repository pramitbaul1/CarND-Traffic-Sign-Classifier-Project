# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 18:03:54 2017

@author: Pramit
"""

# Load pickled data
import pickle
import numpy as np

# TODO: Fill this in based on where you saved the training and testing data

training_file = './datasets/train.p'
validation_file='./datasets/valid.p'
testing_file = './datasets/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
print('Before Grayscale conversion')
print(X_train.shape)
print(X_test.shape)
print(X_valid.shape)
print('After Grayscale conversion')
X_train=np.sum(X_train/3,axis=3,keepdims=True)
X_test=np.sum(X_test/3,axis=3,keepdims=True)
X_valid=np.sum(X_valid/3,axis=3,keepdims=True)
print(X_train.shape)
print(X_test.shape)
print(X_valid.shape)
### Feel free to use as many code cells as needed.

X_train=(X_train-128)/128
X_test=(X_test-128)/128
X_valid=(X_valid-128)/128