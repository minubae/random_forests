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
#         result = sum/n
#         return result
#     else:
#         result = 0
#         return result

def getComparision(numVar, numObs, probability):

    p = numVar
    n = numObs
    prob = probability

    avg_accuracy = []
    sum1 = 0
    sum2 = 0
    N = 100
    for i  in range(N):

        clf1 = 0
        clf2 = 0
        trainX = []
        trainY = []
        testX = []

        trainX = getSynData(p, n, prob)
        testX = getSynData(p, n, prob)
        trainY = getPrediction(trainX)
        # print('Train Prediction: ', trainY)
        # print('\n')
        # tree.DecisionTreeRegressor()
        # tree.DecisionTreeClassifier()
        clf1 = tree.DecisionTreeClassifier()
        clf1 = clf1.fit(trainX,trainY)

        clf2 = RandomForestClassifier(max_depth=4, random_state=0) #max_depth=2, random_state=0
        clf2 = clf2.fit(trainX, trainY)

        testX = getSynData(p, n, prob)
        # print(testData)
        testPredict = getPrediction(testX)
        # print('Test Prediction: ', testPredict)
        treePredict = clf1.predict(testX)
        # print('Tree Prediction: ', treePredict)
        rf_predict = clf2.predict(testX)
        # print('RFor Prediction: ', rf_predict)

        accuracy1 = clf1.score(testX, testPredict)
        sum1 += accuracy1

        accuracy2 = clf2.score(testX, testPredict)
        sum2 += accuracy2

        # print('Tree Accuracy[',i,']:', accuracy1)
        # print('RF Accuracy[',i,']:', accuracy2)
        # print('\n')
        # accuracy = 0
    avg_accuracy1 = sum1/N
    avg_accuracy2 = sum2/N

    avg_accuracy.append(avg_accuracy1)
    avg_accuracy.append(avg_accuracy2)

    print('* Number of Features: ', p)
    print('* Number of Observations: ', n)
    print('* Average of Tree Accuracy: ', avg_accuracy1)
    print('* Average of RF Accuracy: ', avg_accuracy2)

    return avg_accuracy

# print(getComparision(5, 100, 0.5))
# print('\n')
# print(getComparision(5, 1000, 0.5))
# print('\n')
#
# print(getComparision(10, 100, 0.5))
# print('\n')
# print(getComparision(10, 1000, 0.5))
# print('\n')
#
# print(getComparision(20, 100, 0.5))
# print('\n')
# print(getComparision(20, 1000, 0.5))
# print('\n')
#
# print(getComparision(30, 100, 0.5))
# print('\n')
# print(getComparision(30, 1000, 0.5))
# print('\n')

X =[]
p = 2
n = 10
error = 0
for i in range(p):
    x = np.random.randint(1, 4, size=n)
    X.append(x)

X = np.transpose(np.array(X))
print('X: ')
print(X)

# plt.suptitle('Random Numbers')
# plt.plot(X, 'ro')
# plt.axis([0, 3, 0, 3])
# plt.show()
