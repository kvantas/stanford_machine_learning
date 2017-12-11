# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 21:33:19 2017

@author: Konstantinos Vantas
"""

import numpy as np
import matplotlib.pyplot as plt

# ==================== Part 1: Load data ======================================
print("\n Loading data ... \n")
data = np.loadtxt('../data/ex2data1.txt', delimiter=',')
print("First 10 examples from the dataset:")
print(data[1:10,])

# The first two columns contains the exam scores and the third column
# contains the label.
X = np.c_[data[:,[0,1]]]
y = np.c_[data[:,2]]

# ==================== Part 2: Plotting =======================================

# We start the exercise by first plotting the data to understand the 
# the problem we are working with.

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


plotData(X, y, label_x =  'Exam 1 score', label_y = 'Exam 2 score',
         label_pos = 'Admitted', label_neg = 'Not admitted')

# ==================== Part 3: Compute Cost and Gradient ======================

# In this part of the exercise, you will implement the cost and gradient
# for logistic regression. 

def sigmoid(z):
    """
    SIGMOID Compute sigmoid functoon
    J = SIGMOID(z) computes the sigmoid of z.
    """
    return 1/(1 + np.exp(-z))

# Setup the data matrix appropriately, and add ones for the intercept term
m, n = X.shape

# Add intercept term to x
X = np.c_[np.ones(X.shape[0]), X]

# Initialize fitting parameters
initial_theta = np.zeros(X.shape[1])

def costFunction(theta, X, y):
    """
    COSTFUNCTION Compute cost for logistic regression
    J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
    parameter for logistic regression
    """
    
    m = y.size  
    hetta = sigmoid(X.dot(theta)).reshape(y.shape)
    J = 1/m * np.sum(-y * np.log(hetta) - (1-y) * np.log(1-hetta), axis = 0)

    # if cost is nan return Inf
    if np.isnan(J[0]):
        return(np.inf)
    return J[0]

def gradFunction(theta, X, y):
    """
    GRADFUNCTION Compute gradient for logistic regression
    gradFunction = GRADFUNCTION(theta, X, y) computes gradient of the cost
    for logistic regression w.r.t. to the parameters.

    """
    m = y.size  
    hetta = sigmoid(X.dot(theta)).reshape(y.shape)
    grad = 1/m * np.sum((hetta - y)*(X), axis = 0)
    return grad

# Compute and display initial cost and gradient
cost = costFunction(initial_theta, X, y)
grad = gradFunction(initial_theta, X, y)
print('Cost at initial theta (zeros):', str(cost));
print('Gradient at initial theta (zeros): \n');
print(grad)

# ==================== Part 4: Optimizing using BFGS algorithm  ===============

from scipy.optimize import minimize
optim = minimize(fun = costFunction, x0=initial_theta, args=(X,y), 
                 method='BFGS', jac=gradFunction,
                 options={'maxiter':400})
optim.x

# plot data
plotData(X[:,[1,2]], y, 'Exam 1 score', 'Exam 2 score', 'Admitted', 'Not admitted')
x1_min, x1_max = X[:,1].min(), X[:,1].max(),
x2_min, x2_max = X[:,2].min(), X[:,2].max(),
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0],1)), xx1.ravel(), xx2.ravel()].dot(optim.x))
h = h.reshape(xx1.shape)
plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')