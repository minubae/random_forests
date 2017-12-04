import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli,poisson,norm,expon
from random import *
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz

# nobs = 100
# theta = 0.3
# Y = np.random.binomial(1, theta, nobs)

# print(bernoulli.rvs(prob,size = n))
# print(norm.rvs(size = N))
# print(poisson.rvs(1,2,size = N))
# print(expon.rvs(5,size = N))

# Number of Variables (Features)
p = 4
# Number of Observations
n = 20
# probability of Each Bernoulli trial
prob =.5
# Synthetic Data Matrix X
X = []
testX  = []
for i in range(p):
    #x_i is a random variable for each feature
    x_i = bernoulli.rvs(prob,size = n)
    X.append(x_i)

X = np.transpose(np.array(X))

# print('Data Matrix X:')
# print(X)

for j in range(p):
    #x_i is a random variable for each feature
    x_j = bernoulli.rvs(prob,size = n)
    testX.append(x_j)

testX = np.transpose(np.array(testX))


# logical_or function satisfying multiple conditions
def logical_or(inputX):
    x = []
    x = inputX
    mult_or = x[np.where(x==1)]
    result = np.sum(mult_or)

    if result != 0:
        # print('hello')
        return 1
    else:
        return 0

# logical_and function satisfying multiple conditions
def logical_and(inputX):
    x = []
    x = inputX
    n = len(x)
    mult_add = x[np.where(x==1)]
    result = np.sum(mult_add)

    if result == n:
        return 1
    else:
        return 0

# logical_not function satisfying multiple conditions
# logical_xor function satisfying multiple conditions

#Training Data Z
Z = []
Y =[]
for i, x in enumerate(X):
    # print('x: ', x)
    y = logical_or(x)
    # y = logical_and(x)
    # print('y (logical_or): ', y)
    # print('y (logical_and): ', y2)
    Y.append(y)
    # x = np.append(x,y)
    # print('z: ', x)
    # Z.append(x)
    # print('')
testY = []

for j, x in enumerate(testX):
    # print('x: ', x)
    y = logical_or(x)
    # y = logical_and(x)
    # print('y (logical_or): ', y)
    # print('y (logical_and): ', y2)
    testY.append(y)


# Z = np.array(Z)
Y = np.array(Y)
print('Original Data:')
print(X)
print('Train_Predict:')
print(Y)


print('Test Data:')
print(testX)
print('Test_Predict:')
print(testY)

clf = tree.DecisionTreeRegressor()
clf = clf.fit(X,Y)

tree_predict = []
for i in range(10):
    predict = clf.predict(testX)
    tree_predict.append(np.array(predict))
# predict = clf.predict(testX)

tree_predict = np.array(tree_predict)
print('DecisionTreePredict: ')
print(tree_predict)
