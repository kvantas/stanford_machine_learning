# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 10:34:05 2017

@author: Konstantinos Vantas
"""

import numpy as np
import matplotlib.pyplot as plt

# ==================== Part 1: Load data ======================================
print("\n Loading data ... \n")
data = np.loadtxt('../data/ex1data2.txt', delimiter=',')
print("First 10 examples from the dataset:")
print(data[1:10,])

# ================= Part 2: Feature Normalization =============================

def featureNormalize(X):
    """
    FEATURENORMALIZE(X) returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when
    working with learning algorithms.
    """
    
    mu = X.mean(axis = 0, keepdims = True)
    sigma = X.std(axis = 0, keepdims = True)
    
    Xnorm = (X - mu) / sigma
    
    return Xnorm, mu, sigma

# Scale features and set them to zero mean
X = np.c_[data[:,[0,1]]]
y = np.c_[data[:,2]]

print(" \n Normalizing Features ...")
[Xnorm, mu, sigma] = featureNormalize(X)

# Add intercept term to X
Xnorm = np.c_[np.ones(Xnorm.shape[0]), Xnorm]

# ================= Part 3: Gradient Descent ==================================

print("\n Running gradient descent ...")

# Choose some alpha value
alpha = 0.01
num_iters = 400

# Compute cost for linear regression
def computeCost(X, y, theta):
    """
    COMPUTECOST Compute cost for linear regression
    J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y
    """
    
    m = y.size
    J = 1 / (2 * m) * np.sum((X.dot(theta) - y)**2)
    
    return(J)

# Perform gradient descent to learn theta
def gradientDescent(X, y, theta, alpha, num_iters):
    """
    GRADIENTDESCENT Performs gradient descent to learn theta
    theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
    taking num_iters gradient steps with learning rate alpha
    """
    
    m = y.size
    J_history = np.zeros([num_iters, 1])
    
    for i in np.arange(num_iters):
        h = X.dot(theta)
        theta = theta - alpha * 1 / m * (X.T.dot(h - y))
        J_history[i] = computeCost(X, y, theta)
    
    return theta, J_history

# Init Theta and Run Gradient Descent 
theta = np.zeros([3,1])
[theta, J_history] = gradientDescent(Xnorm, y, theta, alpha, num_iters)

# Plot the convergence graph
plt.plot(J_history)
plt.ylabel('Cost J')
plt.xlabel('Number of iterations')
plt.show()

# Display gradient descent's result
print('Theta computed from gradient descent: \n');
print(theta);

# Estimate the price of a 1650 sq-ft, 3 br house
# Recall that the first column of X is all-ones. Thus, it does not need to be 
# normalized.

Xnew = np.array([1, (1650 - mu[0,0]) / sigma[0,0], 3 - mu[0,1]/sigma[0,1]])
price = Xnew.dot(theta).round(2)[0]

print("Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): \n$"
      + str(price))

# ================= Part 4: Selecting learning rates ==========================

# Choose alpha value
alpha_v = [1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
num_iters = 50

for a in alpha_v:
    theta = np.zeros([3,1])
    [theta, J_history] = gradientDescent(Xnorm, y, theta, a, num_iters)
    plt.plot(J_history, label = a)
plt.legend(loc = 4)
plt.show()

# ================= Part 5: Normal Equations ==================================

print("Solving with normal equations ... \n")

# Compute the closed-form solution to linear regression
def normalEqn(X, y):
    """
    NORMALEQN(X,y) computes the closed-form solution to linear 
    regression using the normal equations.
    """
    
    tmp =  X.T.dot(X)
    tmp = np.linalg.inv(tmp)
    
    theta = tmp.dot(X.T).dot(y)
    return(theta)

theta_norm = normalEqn(Xnorm, y)

# Display normal equations's result
print('Theta computed from normal equations: \n');
print(theta_norm);

# Estimate the price of a 1650 sq-ft, 3 br house
price_norm = Xnew.dot(theta_norm).round(2)[0]
print("Predicted price of a 1650 sq-ft, 3 br house (using closed-form solution): \n$"
      + str(price_norm))

# ================= Part 6: sklearn implementation ============================

from sklearn.linear_model import LinearRegression

# Fitting Multiple Linear Regression 
regressor = LinearRegression(fit_intercept=True, normalize=True)
regressor.fit(X, y)

theta_sk = np.array([regressor.intercept_[0], 
                    regressor.coef_[0,0],
                    regressor.coef_[0,1]]).reshape(3,1)
    
# Display sklearn's result
print('Theta computed from sklearn LinearRegression: \n');
print(theta_sk);

# Estimate the price of a 1650 sq-ft, 3 br house
price_sk = regressor.predict(np.array([1650, 3]).reshape(1,2))
price_sk = price_sk.round(2)
 
print("Predicted price of a 1650 sq-ft, 3 br house (using sklearn): \n$"
      + str(price_sk[0,0]))