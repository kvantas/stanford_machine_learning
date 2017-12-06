# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 15:06:51 2017

@author: Konstantinos Vantas
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# ==================== Part 1: Load data ======================================
print("\n Loading data ... \n")
data = np.loadtxt('../data/ex1data1.txt', delimiter=',')
print(data[1:5,])

# ==================== Part 2: Plot data ======================================
print("\n Plotting Data ... \n")

plt.scatter(x = data[:,0], y = data[:,1], 
            s = 30, c = 'r', marker = 'x', linewidths=1)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()

# =================== Part 3: Gradient descent ================================
print("\n Running Gradient Descent ... \n")

# Add a column of ones to x
X = np.c_[np.ones(data.shape[0]), data[:,0]]
y = np.c_[data[:,1]]

# initialize fitting parameters
theta = np.zeros([2,1])

# Some gradient descent settings
iterations = 1500;
alpha = 0.01;

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

# compute and display initial cost
print("Initial cost = " + str(computeCost(X, y, theta)))

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
    
# run gradient descent
[theta,  J_history] = gradientDescent(X, y, theta, alpha, iterations);
print('Theta found by gradient descent: \n' + str(theta));

# Plot the linear fit
xx = np.arange(5,23)
yy = theta[0]+theta[1]*xx

plt.scatter(x = data[:,0], y = data[:,1], 
            s = 30, c = 'r', marker = 'x', linewidths=1)
plt.plot(xx,yy, label='Linear regression (Gradient descent)')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.legend(loc=4)
plt.show()

# Predict values for population sizes of 35,000 and 70,000
predict1 = 1 * theta[0] + 3.5 * theta[1] 
print('For population = 35,000, we predict a profit of ' + 
      str(predict1 * 10000))

predict2 = 1 * theta[0] + 7.0 * theta[1] 
print('For population = 70,000, we predict a profit of ' + 
      str(predict2 * 10000))

# ============= Part 4: Visualizing J(theta_0, theta_1) =======================

# plot cost J
plt.plot(J_history)
plt.ylabel('Cost J')
plt.xlabel('Iterations')
plt.show()



# Grid over which we will calculate J
B0 = np.linspace(-10, 10, 100)
B1 = np.linspace(-1, 4, 100)
xx, yy = np.meshgrid(B0, B1, indexing='xy')

# initialize J_vals to a matrix of 0'
Z = np.zeros((B0.size,B1.size))

# Calculate Z-values (Cost) based on grid of coefficients
for (i,j),v in np.ndenumerate(Z):
    Z[i,j] = computeCost(X,y, theta=[[xx[i,j]], [yy[i,j]]])

fig = plt.figure(figsize=(15,6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection='3d')

# Left plot
CS = ax1.contour(xx, yy, Z, np.logspace(-2, 3, 20), cmap=plt.cm.jet)
ax1.scatter(theta[0],theta[1], c='r')

# Right plot
ax2.plot_surface(xx, yy, Z, rstride=1, cstride=1, alpha=0.6, cmap=plt.cm.jet)
ax2.set_zlabel('Cost')
ax2.set_zlim(Z.min(),Z.max())
ax2.view_init(elev=15, azim=230)

# settings common to both plots
for ax in fig.axes:
    ax.set_xlabel(r'$\theta_0$', fontsize=17)
    ax.set_ylabel(r'$\theta_1$', fontsize=17)

