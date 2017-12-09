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
# p = 10
# Number of Observations
# n = 100
# probability of Each Bernoulli trial
# prob =.5

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

    Y = np.array(Y)
    return Y

def getAccuracy(Prediction_1, Prediction_2):

    result = 0
    coincidence = 0
    testY = Prediction_1
    treeY = Prediction_2

    n = len(testY)

    for i in range(n):
        if(testY[i] == treeY[i]):
            coincidence += 1

    result = coincidence/n
    return result

def getRandNumSynData(Range, Features, Observations):
    X =[]
    p = Features
    n = Observations
    ran = Range
    error = 0
    for i in range(p):
        x = np.random.randint(1, ran, size=n)
        X.append(x)

    X = np.transpose(np.array(X))
    return X

def getRandNumPrediction2(TrainData):

    trainData = TrainData

    trainPrediction = []
    for i, x in enumerate(trainData):

        if x[0]==1 and x[1]==1:
            y = 3
            trainPrediction.append(y)

        if x[0]==1 and x[1]==2:
            y = 5
            trainPrediction.append(y)

        if x[0]==1 and x[1]==3:
            y = 4
            trainPrediction.append(y)

        if x[0]==2 and x[1]==1:
            y = 3
            trainPrediction.append(y)

        if x[0]==2 and x[1]==2:
            y = 5
            trainPrediction.append(y)

        if x[0]==2 and x[1]==3:
            y = np.random.randint(1, 3)
            trainPrediction.append(y)

        if x[0]==3 and x[1]==1:
            y = 3
            trainPrediction.append(y)

        if x[0]==3 and x[1]==2:
            y = 6
            trainPrediction.append(y)

        if x[0]==3 and x[1]==3:
            y = 1
            trainPrediction.append(y)

    trainPrediction = np.array(trainPrediction)
    return trainPrediction

def getRandNumPrediction3(TrainData):

    trainData = TrainData

    trainPrediction = []
    for i, x in enumerate(trainData):

        if x[0]==1 and x[1]==1 and x[2] == 1:
            # y = np.random.randint(3, 6)
            y = 1
            trainPrediction.append(y)
        if x[0]==1 and x[1]==1 and x[2] == 2:
            # y = np.random.randint(3, 6)
            y = 2
            trainPrediction.append(y)
        if x[0]==1 and x[1]==1 and x[2] == 3:
            # y = np.random.randint(3, 6)
            y = 3
            trainPrediction.append(y)


        if x[0]==1 and x[1]==2 and x[2] == 1:
            # y = np.random.randint(1, 3)
            y = 3
            trainPrediction.append(y)
        if x[0]==1 and x[1]==2 and x[2] == 2:
            # y = np.random.randint(1, 3)
            y = 4
            trainPrediction.append(y)
        if x[0]==1 and x[1]==2 and x[2] == 3:
            # y = np.random.randint(1, 3)
            y = 2
            trainPrediction.append(y)


        if x[0]==1 and x[1]==3 and x[2] == 1:
            y = np.random.randint(4, 6)
            trainPrediction.append(y)
        if x[0]==1 and x[1]==3 and x[2] == 2:
            y = np.random.randint(4, 6)
            trainPrediction.append(y)
        if x[0]==1 and x[1]==3 and x[2] == 3:
            y = np.random.randint(4, 6)
            trainPrediction.append(y)

        if x[0]==2 and x[1]==1 and x[2] == 1:
            # y = np.random.randint(1, 4)
            y = 2
            trainPrediction.append(y)
        if x[0]==2 and x[1]==1 and x[2] == 2:
            # y = np.random.randint(1, 4)
            y = 3
            trainPrediction.append(y)
        if x[0]==2 and x[1]==1 and x[2] == 3:
            # y = np.random.randint(1, 4)
            y =1
            trainPrediction.append(y)


        if x[0]==2 and x[1]==2 and x[2] == 1:
            y = np.random.randint(3, 5)
            trainPrediction.append(y)
        if x[0]==2 and x[1]==2 and x[2] == 2:
            y = np.random.randint(3, 5)
            trainPrediction.append(y)
        if x[0]==2 and x[1]==2 and x[2] == 3:
            y = np.random.randint(3, 5)
            trainPrediction.append(y)


        if x[0]==2 and x[1]==3 and x[2] == 1:
            # y = np.random.randint(1, 4)
            y = 5
            trainPrediction.append(y)
        if x[0]==2 and x[1]==3 and x[2] == 2:
            # y = np.random.randint(1, 4)
            y = 4
            trainPrediction.append(y)
        if x[0]==2 and x[1]==3 and x[2] == 3:
            # y = np.random.randint(1, 4)
            y = 2
            trainPrediction.append(y)

        if x[0]==3 and x[1]==1 and x[2] == 1:
            # y = np.random.randint(2, 4)
            y = 3
            trainPrediction.append(y)
        if x[0]==3 and x[1]==1 and x[2] == 2:
            # y = np.random.randint(2, 4)
            y = 4
            trainPrediction.append(y)
        if x[0]==3 and x[1]==1 and x[2] == 3:
            # y = np.random.randint(2, 4)
            y = 1
            trainPrediction.append(y)

        if x[0]==3 and x[1]==2 and x[2] == 1:
            # y = np.random.randint(5, 7)
            y = 1
            trainPrediction.append(y)
        if x[0]==3 and x[1]==2 and x[2] == 2:
            # y = np.random.randint(5, 7)
            y = 2
            trainPrediction.append(y)
        if x[0]==3 and x[1]==2 and x[2] == 3:
            # y = np.random.randint(5, 7)
            y = 4
            trainPrediction.append(y)

        if x[0]==3 and x[1]==3 and x[2] == 1:
            y = np.random.randint(1, 3)
            trainPrediction.append(y)
        if x[0]==3 and x[1]==3 and x[2] == 2:
            # y = np.random.randint(1, 3)
            y = 5
            trainPrediction.append(y)
        if x[0]==3 and x[1]==3 and x[2] == 3:
            y = np.random.randint(1, 3)
            trainPrediction.append(y)

    trainPrediction = np.array(trainPrediction)
    return trainPrediction


def getTreeAccuracy(Range, Features, Observations, NumSimulations):

    accuracy = 0
    sum_accuracy = 0
    avg_accuracy = 0

    ran = Range
    N = NumSimulations
    features = Features
    observations = Observations

    for i in range(N):

        trainX = getRandNumSynData(ran, features, observations)
        trainY = getRandNumPrediction2(trainX)

        clf1 = tree.DecisionTreeClassifier()
        clf1 = clf1.fit(trainX,trainY)

        testX = getRandNumSynData(ran, features, observations)
        testY = getRandNumPrediction2(testX)

        treeY = clf1.predict(testX)
        # accuracy = clf1.score(testX, testY)
        accuracy = getAccuracy(testY, treeY)

        sum_accuracy += accuracy

    avg_accuracy = sum_accuracy/N

    return avg_accuracy


def getRfAccuracy(Range, Features, Observations, NumSimulations):

    accuracy = 0
    sum_accuracy = 0
    avg_accuracy = 0

    ran = Range
    N = NumSimulations
    features = Features
    observations = Observations

    for i in range(N):

        trainX = getRandNumSynData(ran, features, observations)
        trainY = getRandNumPrediction2(trainX)

        clf2 = RandomForestClassifier(max_depth=4, random_state=0) #max_depth=2, random_state=0
        clf2 = clf2.fit(trainX, trainY)

        testX = getRandNumSynData(ran, features, observations)
        testY = getRandNumPrediction2(testX)

        rf_predict = clf2.predict(testX)
        # accuracy = clf2.score(testX, testY)
        accuracy = getAccuracy(testY, rf_predict)

        sum_accuracy += accuracy

    avg_accuracy = sum_accuracy/N

    return avg_accuracy

ran = 4
features = 2
observations = 100
num_simulations = 100
treeAccuracy = getTreeAccuracy(ran, features, observations, num_simulations)
rfAccuracy = getRfAccuracy(ran, features, observations, num_simulations)

print('• Number of Features: ',features)
print('• Number of Observations: ', observations)
print('• Tree Accuracy: ', treeAccuracy)
print('• Rfor Accuracy: ', rfAccuracy)
