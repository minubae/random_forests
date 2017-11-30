import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli,poisson,norm,expon
from random import *

'''
def getSpikeData(length, fireRate):

    T = 0
    coin = 0
    fRate = 0
    binprob = 0
    spikeData = []

    #length of train
    T = length;
    # initialize the Spike Train
    spikeData = np.zeros(T)
    fRate = fireRate
    # binprob = (1./T)*fRate
    binprob = 0.5
    # print('binprob:', binprob)
    for k in range(0,int(T)):
        coin = np.random.uniform()
        # print('Coin:', coin)
        if coin <= binprob:
            spikeData[k] = 1

    return spikeData


print(getSpikeData(10, 40))
'''
'''
n, p = 1, 0.5 # number of trials, probability of each trial
s = np.random.binomial(n, p, 20)
print(s)
'''
prob =.5
n = 10
print(bernoulli.rvs(prob,size = n))
# print(norm.rvs(size = N))
# print(poisson.rvs(1,2,size = N))
# print(expon.rvs(5,size = N))

p = 3
for i in range(p):
    
