import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli,poisson,norm,expon
from random import *

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

X = np.array(X)

print('Data Matrix X:')
print(X)
