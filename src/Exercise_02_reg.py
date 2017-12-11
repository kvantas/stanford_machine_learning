# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 17:20:37 2017

@author: Konstantinos Vantas
"""

import numpy as np
import matplotlib.pyplot as plt

# ==================== Part 1: Load data ======================================
print("\n Loading data ... \n")
data = np.loadtxt('../data/ex2data2.txt', delimiter=',')
print("First 10 examples from the dataset:")
print(data[1:10,])

X = np.c_[data[:,[0,1]]]
y = np.c_[data[:,2]]

def plotData(X, y, label_x, label_y, label_pos, label_neg):
    """
    PLOTDATA Plots the data points X and y into a new figure 
    PLOTDATA(x,y) plots the data points with + for the positive examples
    and o for the negative examples. X is assumed to be a Mx2 numpy array.

    """
    # Get indexes for class 0 and class 1
    neg = np.squeeze(y == 0)
    pos = np.squeeze(y == 1)
    
    # plot examples
    axes = plt.gca()
    axes.scatter(x = X[pos, 0], y = X[pos, 1], c= 'black', marker= '+', s=60,
                linewidth=3, label=label_pos)
    axes.scatter(x = X[neg, 0], y = X[neg, 1], c='y', s=60, marker = 'o',
                label=label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon= True, fancybox = True)


plotData(X, y, label_x =  'Microchip Test 1', label_y = 'Microchip Test 2',
         label_pos = 'accepted', label_neg = 'rejected')

# ==================== Part 2: Regularized Logistic Regression ================

#  In this part, you are given a dataset with data points that are not
#  linearly separable. However, you would still like to use logistic 
#  regression to classify the data points. 
#
#  To do so, you introduce more features to use -- in particular, you add
#  polynomial features to our data matrix (similar to polynomial
#  regression).

def mapFeature(X1, X2, degree = 6):
    # MAPFEATURE Feature mapping function to polynomial features

   # MAPFEATURE(X1, X2) maps the two input features
   # to quadratic features used in the regularization exercise.

   # Returns a new feature array with more features, comprising of 
   # X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..

   # Inputs X1, X2 must be the same size
   out = np.ones(X1.shape[0])
   
   for i in range(1, degree+1):
       for j in range(0, i+1):
           out = np.c_[out, (X1**(i-j)) * (X2**j)]
   
   return out;

X = mapFeature(X[:, 0], X[:, 1]);

# Initialize fitting parameters
initial_theta = np.zeros([X.shape[1],1])

# Set regularization parameter lambda to 1
Lambda = 1

def sigmoid(z):
    """
    SIGMOID Compute sigmoid functoon
    J = SIGMOID(z) computes the sigmoid of z.
    """
    return 1/(1 + np.exp(-z))

def costFunction(theta, X, y, Lambda):
    """
    COSTFUNCTION Compute cost for logistic regression with regularization
    J = COSTFUNCTION(theta, X, y, lambda) computes the cost of using
    theta as the parameter for regularized logistic regression
    """
    
    m = y.size  
    hetta = sigmoid(X.dot(theta)).reshape(y.shape)
    J = 1/m * np.sum(-y * np.log(hetta) - (1-y) * np.log(1-hetta), axis = 0) + Lambda/(2*m) * np.sum(theta[1:, ]**2, axis = 0)

    # if cost is nan return Inf
    if np.isnan(J[0]):
        return(np.inf)
    return J[0]

def gradFunction(theta, X, y, Lambda):
    
    """
    GRADFUNCTION Compute gradient for logistic regression
    gradFunction = GRADFUNCTION(theta, X, y) computes gradient of the cost
    for logistic regression w.r.t. to the parameters.

    """
    m = y.size
    
    hetta = sigmoid(X.dot(theta)).reshape(y.shape)
    grad = 1/m * np.sum((hetta - y)*(X), axis = 0, keepdims= True).T + Lambda/m * np.r_[[[0]],theta[1:].reshape(-1,1)]
    return grad.flatten()

# Compute and display initial cost and gradient
cost = costFunction(initial_theta, X, y, Lambda)
print('Cost at initial theta (zeros):', str(cost))

def predict(theta, X):
    return sigmoid(X.dot(theta.T))

# ==================== Part 3: Optimizing using BFGS algorithm  ===============

from scipy.optimize import minimize
#optim = minimize(fun = costFunction, x0=initial_theta, args=(X, y, Lambda), 
#                 method='BFGS', jac=gradFunction,
#                 options={'maxiter':400})
#optim.fun

# plot data and decision boundary for various values of lambda
for Lambda in [0, 0.1, 1, 10, 100]:
    print('l = ', str(Lambda))
    optim = minimize(fun = costFunction, x0=initial_theta, args=(X,y, Lambda), 
                 method='BFGS', jac=gradFunction,
                 options={'maxiter':400})
    # Accuracy
    accuracy = (100*sum( (predict(optim.x, X) >= 0.5) == y.ravel())/y.size)
    
    # create grid for desicion boundary
    x1_min, x1_max = X[:,1].min(), X[:,1].max()
    x2_min, x2_max = X[:,2].min(), X[:,2].max()
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    
    plotData(data[:,[0,1]], y, label_x =  'Microchip Test 1', label_y = 'Microchip Test 2',
         label_pos = 'accepted', label_neg = 'rejected')
    h = sigmoid(mapFeature(xx1.ravel(), xx2.ravel()).dot(optim.x))
    h = h.reshape(xx1.shape)
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')
    plt.title('Train accuracy {}% with Lambda = {}'.format(np.round(accuracy, decimals=2), Lambda))
    plt.show()