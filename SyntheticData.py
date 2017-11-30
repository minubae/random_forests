import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli,poisson,norm,expon
from random import *

# nobs = 100
# theta = 0.3
# Y = np.random.binomial(1, theta, nobs)

# print(bernoulli.rvs(prob,size = n))
# print(norm.rvs(size = N))
# print(poisson.rvs(1,2,size = N))
# print(expon.rvs(5,size = N))

# Number of Variables (Features)
p = 3
# Number of Observations
n = 10
# probability of Each Bernoulli trial
prob =.5
# Synthetic Data Matrix X
X = []

for i in range(p):
    #x_i is a random variable for each feature
    x_i = bernoulli.rvs(prob,size = n)
    X.append(x_i)

X = np.transpose(np.array(X))

print('Data Matrix X:')
print(X)

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

for i, x in enumerate(X):

    y = logical_or(x)
    print(y)
