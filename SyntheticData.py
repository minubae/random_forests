import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli,poisson,norm,expon
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
# import graphviz

# nobs = 100
# theta = 0.3
# Y = np.random.binomial(1, theta, nobs)

# print(bernoulli.rvs(prob,size = n))
# print(norm.rvs(size = N))
# print(poisson.rvs(1,2,size = N))
# print(expon.rvs(5,size = N))

# Number of Variables (Features)
p = 10
# Number of Observations
n = 100
# probability of Each Bernoulli trial
prob =.5
# Synthetic Data Matrix X
TrainX = []
TestX = []

def getSynData(P, N, Prob):
    p = P
    n = N
    prob = Prob
    trainX  = []
    for j in range(p):
        #x_i is a random variable for each feature
        x_j = bernoulli.rvs(prob,size = n)
        trainX.append(x_j)

    trainX = np.transpose(np.array(trainX))
    return trainX

TrainX = getSynData(p, n, prob)
TestX = getSynData(p, n, prob)

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
def getDecision(inputX, prob):

    x = []
    x = inputX
    n = len(x)

    criterion = n*prob

    mult_add = x[np.where(x==1)]
    result = np.sum(mult_add)

    if result > criterion:
        return 1
    else:
        return 0

def getPrediction(Data):
    X = Data
    Y =[]
    for i, x in enumerate(X):
        # print('x: ', x)
        # y = logical_or(x)
        y = getDecision(x, 0.5)
        # y = logical_and(x)
        # print('y (logical_or): ', y)
        # print('y (logical_and): ', y2)
        Y.append(y)
        # x = np.append(x,y)
        # print('z: ', x)
        # Z.append(x)
        # print('')
    Y = np.array(Y)
    return Y

# def getAccuracy(TestY, TreeY):
#     sum = 0
#     result = 0
#     testY = TestY
#     treeY = TreeY
#
#     # n = len(testY)
#     for i in range(len(testY)):
#         if(testY[i] == treeY[i]):
#             sum+=1
#
#     mult_num = len(testY[np.where(testY==1)])
#     print('mult_num: ', mult_num)
#     if mult_num != 0:
#         result = sum/mult_num
#         return result
#     else:
#         result = 0
#         return result

TrainY = []
TrainY = getPrediction(TrainX)

print('Train Prediction: ', TrainY)
print('\n')
'''
TestY = []
TestY = getPrediction(TestX)
print('Original Data:')
print(TrainX)
print('Train_Predict:')
print(TrainY)

print('Test Data:')
print(TestX)
print('Test_Predict:')
print(TestY)

print('Tree_Predict: ')
print(TreeY)
print('Accuracy: ', getAccuracy(TestY, TreeY))
'''

# tree.DecisionTreeRegressor()
# tree.DecisionTreeClassifier()
clf1 = tree.DecisionTreeClassifier()
clf1 = clf1.fit(TrainX,TrainY)
# TreeY = []
# TreeY = clf.predict(TestX)
clf2 = RandomForestClassifier(max_depth=4, random_state=0) #max_depth=2, random_state=0
clf2 = clf2.fit(TrainX, TrainY)


sum1 = 0
sum2 = 0
N = 100
for i  in range(N):
    testData = getSynData(p, n, prob)
    # print(testData)
    testPredict = getPrediction(testData)
    print('Test Prediction: ', testPredict)
    treePredict = clf1.predict(testData)
    print('Tree Prediction: ', treePredict)
    rf_predict = clf2.predict(testData)
    print('RFor Prediction: ', rf_predict)

    accuracy1 = clf1.score(testData, testPredict)
    sum1 += accuracy1

    accuracy2 = clf2.score(testData, testPredict)
    sum2 += accuracy2

    print('Tree Accuracy[',i,']:', accuracy1)
    print('RF Accuracy[',i,']:', accuracy2)
    print('\n')
    # accuracy = 0
avg_accuracy1 = sum1/N
avg_accuracy2 = sum2/N
print('* Number of Features: ', p)
print('* Number of Observations: ', n)
print('* Average of Tree Accuracy: ', avg_accuracy1)
print('* Average of RF Accuracy: ', avg_accuracy2)
