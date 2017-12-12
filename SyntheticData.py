import numpy as np
import collections
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
        x_j = bernoulli.rvs(prob, size=n)
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
    # error = 0
    X = np.random.randint(1, ran, size=(n, p))
    return X


def getRandNumPrediction2(Range, TrainData):

    temp = []; index = []
    prediction = []; dictY = {}

    ran = Range
    data = TrainData
    y = 1

    for i in range(1, ran):
        for j in range(1, ran):

            temp.append(i)
            temp.append(j)

            for inx, x in enumerate(data):
                if np.array_equal(x, temp):
                    # dictY[inx]=y
                    dictY[inx] = y

            temp = []
            y += 1

    prediction = collections.OrderedDict(sorted(dictY.items()))
    prediction = prediction.values()
    output = []
    for i in prediction:
        output.append(i)

    output = np.array(output)
    return output

'''
def getRandNumPrediction2(Range, TrainData):

    trainData = TrainData
    ran = Range
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
'''
def getRandNumPrediction3(Range, TrainData):

    trainData = TrainData
    trainPrediction = []
    ran = Range

    for i, x in enumerate(trainData):

        # if x[0]==1 and x[2] == 1:
        if x[0]==1 and x[1]==1 and x[2] == 1:
            # y = np.random.randint(3, 6)
            y = 1
            trainPrediction.append(y)
        # if x[0]==1 and x[1]==1:
        if x[0]==1 and x[1]==1 and x[2] == 2:
            # y = np.random.randint(3, 6)
            y = 2
            trainPrediction.append(y)
        # if x[1]==1 and x[2] == 3:
        if x[0]==1 and x[1]==1 and x[2] == 3:
            # y = np.random.randint(3, 6)
            y = 3
            trainPrediction.append(y)


        # if x[0]==1 and x[2] == 1:
        if x[0]==1 and x[1]==2 and x[2] == 1:
            # y = np.random.randint(1, 3)
            y = 4
            trainPrediction.append(y)
        # if x[0]==1 and x[1]==2:
        if x[0]==1 and x[1]==2 and x[2] == 2:
            # y = np.random.randint(1, 3)
            y = 5
            trainPrediction.append(y)
        # if x[1]==2 and x[2] == 3:
        if x[0]==1 and x[1]==2 and x[2] == 3:
            # y = np.random.randint(1, 3)
            y = 6
            trainPrediction.append(y)


        # if x[0]==1 and x[2] == 1:
        if x[0]==1 and x[1]==3 and x[2] == 1:
            y = 7
            # y = np.random.randint(4, 6)
            trainPrediction.append(y)
        if x[0]==1 and x[1]==3 and x[2] == 2:
            y = 8
            # y = np.random.randint(4, 6)
            trainPrediction.append(y)
        if x[0]==1 and x[1]==3 and x[2] == 3:
            y = 9
            # y = np.random.randint(4, 6)
            trainPrediction.append(y)

        if x[0]==2 and x[1]==1 and x[2] == 1:
            # y = np.random.randint(1, 4)
            y = 10
            trainPrediction.append(y)
        if x[0]==2 and x[1]==1 and x[2] == 2:
            # y = np.random.randint(1, 4)
            y = 11
            trainPrediction.append(y)
        if x[0]==2 and x[1]==1 and x[2] == 3:
            # y = np.random.randint(1, 4)
            y = 12
            trainPrediction.append(y)


        if x[0]==2 and x[1]==2 and x[2] == 1:
            y = 13
            # y = np.random.randint(3, 5)
            trainPrediction.append(y)
        if x[0]==2 and x[1]==2 and x[2] == 2:
            y = 14
            # y = np.random.randint(3, 5)
            trainPrediction.append(y)
        if x[0]==2 and x[1]==2 and x[2] == 3:
            y = 15
            # y = np.random.randint(3, 5)
            trainPrediction.append(y)


        if x[0]==2 and x[1]==3 and x[2] == 1:
            # y = np.random.randint(1, 4)
            y = 16
            trainPrediction.append(y)
        if x[0]==2 and x[1]==3 and x[2] == 2:
            # y = np.random.randint(1, 4)
            y = 17
            trainPrediction.append(y)
        if x[0]==2 and x[1]==3 and x[2] == 3:
            # y = np.random.randint(1, 4)
            y = 18
            trainPrediction.append(y)


        if x[0]==3 and x[1]==1 and x[2] == 1:
            y = np.random.randint(18, 21)
            # y = 19
            trainPrediction.append(y)
        if x[0]==3 and x[1]==1 and x[2] == 2:
            y = np.random.randint(18, 21)
            # y = 20
            trainPrediction.append(y)
        if x[0]==3 and x[1]==1 and x[2] == 3:
            y = np.random.randint(18, 21)
            # y = 21
            trainPrediction.append(y)


        if x[0]==3 and x[1]==2 and x[2] == 1:
            # y = np.random.randint(5, 7)
            y = 22
            trainPrediction.append(y)
        if x[0]==3 and x[1]==2 and x[2] == 2:
            # y = np.random.randint(5, 7)
            y = 23
            trainPrediction.append(y)
        if x[0]==3 and x[1]==2 and x[2] == 3:
            # y = np.random.randint(5, 7)
            y = 24
            trainPrediction.append(y)


        if x[0]==3 and x[1]==3 and x[2] == 1:
            # y = np.random.randint(1, 3)
            y = 25
            trainPrediction.append(y)
        if x[0]==3 and x[1]==3 and x[2] == 2:
            # y = np.random.randint(1, 3)
            y = 26
            trainPrediction.append(y)
        if x[0]==3 and x[1]==3 and x[2] == 3:
            y = 27
            # y = np.random.randint(1, 3)
            trainPrediction.append(y)

    trainPrediction = np.array(trainPrediction)
    return trainPrediction

def getRandNumAccuracy(Range, Features, Observations, NumSimulations):

    tree_accuracy = 0
    rf_accuracy = 0

    tree_sum_accuracy = 0
    rf_sum_accuracy = 0

    avg_accuracy = []
    trainX = []; trainY = []
    testX = []; testY = []

    ran = Range
    N = NumSimulations
    features = Features
    observations = Observations

    for i in range(N):

        trainX = getRandNumSynData(ran, features, observations)
        trainY = getRandNumPrediction2(ran, trainX)

        # print(trainX)
        # print(trainY)
        # break

        clf1 = tree.DecisionTreeClassifier()
        clf1 = clf1.fit(trainX,trainY)

        clf2 = RandomForestClassifier(max_depth=6, random_state=0) #max_depth=2, random_state=0
        clf2 = clf2.fit(trainX, trainY)

        testX = getRandNumSynData(ran, features, observations)
        testY = getRandNumPrediction2(ran, testX)

        # print(trainX)
        # print('Train Y: ')
        # print(trainY)
        # break

        treeY = clf1.predict(testX)
        rf_predict = clf2.predict(testX)

        # print(treeY)
        # print(rf_predict)

        # accuracy = clf1.score(testX, testY)
        # accuracy = clf2.score(testX, testY)
        tree_accuracy = getAccuracy(testY, treeY)
        rf_accuracy = getAccuracy(testY, rf_predict)

        tree_sum_accuracy += tree_accuracy
        rf_sum_accuracy += rf_accuracy

        trainX = []; trainY = []
        testX = []; testY = []


    tree_avg_accuracy = tree_sum_accuracy/N
    rf_avg_accuracy = rf_sum_accuracy/N

    avg_accuracy.append(tree_avg_accuracy)
    avg_accuracy.append(rf_avg_accuracy)

    avg_accuracy = np.array(avg_accuracy)

    return avg_accuracy

ran = 4
features = 2
observations = 50
num_simulations = 100
accuracies = getRandNumAccuracy(ran, features, observations, num_simulations)

print('• Number of Features: ',features)
print('• Number of Observations: ', observations)
print('• Tree Accuracy: ', accuracies[0])
print('• Rfor Accuracy: ', accuracies[1])

def getSynLinearDataSet2(Slope, Observations, Error):

    error = Error
    slope = Slope
    N = Observations

    dataX = []
    dataY = []
    X1 = []
    X2 = []

    for i in range(N):
        x1 = i/N
        X1.append(x1)
    dataX.append(X1)

    for i, x in enumerate(X1):
        e = np.random.uniform(-error,error)
        x2 = slope*x + e
        X2.append(x2)

    dataX.append(X2)
    dataX = np.transpose(np.array(dataX))

    return dataX

def getSynLinearDataSet3(Slope1, Slope2, Observations, Error):

    error = Error
    slope1 = Slope1
    slope2 = Slope2
    N = Observations

    dataX = []
    dataY = []
    X1 = []
    X2 = []
    X3 = []

    for i in range(N):
        x1 = i/N
        X1.append(x1)
    dataX.append(X1)

    for i, x in enumerate(X1):
        e = np.random.uniform(-error,error)
        x2 = slope1*x
        X2.append(x2)

    dataX.append(X2)

    for i, x in enumerate(X2):
        e = np.random.uniform(-error,error)
        x3 = slope2*x + e
        X3.append(x3)
    dataX.append(X3)
    dataX = np.transpose(np.array(dataX))

    return dataX

def getLinearDataPrediction2(Slope, DataX):

    slope = Slope
    dataX = DataX
    dataY = []

    for x in dataX:
        if x[1] > slope*x[0]:
            y = 1
            dataY.append(y)
        else:
            y = 0
            dataY.append(y)

    dataY = np.array(dataY)

    return dataY

def getLinearDataPrediction3(Slope1,Slope2, DataX):

    slope1 = Slope1
    slope2 = Slope2
    dataX = DataX
    dataY = []

    for x in dataX:
        if x[2] > slope1*x[0]+slope2*x[1]:
            y = 1
            dataY.append(y)
        else:
            y = 0
            dataY.append(y)

    dataY = np.array(dataY)

    return dataY


def getAccuracyLinearData(Slope1, Slope2, Observations, Error, NumSimulations):

    tree_accuracy = 0
    rf_accuracy = 0

    tree_sum_accuracy = 0
    rf_sum_accuracy = 0

    tree_avg_accuracy = 0
    rf_avg_accuracy = 0

    avg_accuracy = []
    trainX = []; trainY = []
    testX = []; testY = []

    slope1 = Slope1
    slope2 = Slope2
    error = Error

    N = NumSimulations
    observations = Observations

    for i in range(N):

        trainX = getSynLinearDataSet2(slope1, observations, error)
        trainY = getLinearDataPrediction2(slope1, trainX)

        testX = getSynLinearDataSet2(slope1, observations, error)
        testY = getLinearDataPrediction2(slope1, testX)

        # trainX = getSynLinearDataSet3(slope1, slope2, observations, error)
        # trainY = getLinearDataPrediction3(slope1, slope2, trainX)
        #
        # testX = getSynLinearDataSet3(slope1, slope2, observations, error)
        # testY = getLinearDataPrediction3(slope1, slope2, testX)

        clf1 = tree.DecisionTreeClassifier()
        clf1 = clf1.fit(trainX,trainY)

        clf2 = RandomForestClassifier(max_depth=6, random_state=0) #max_depth=2, random_state=0
        clf2 = clf2.fit(trainX, trainY)

        treeY = clf1.predict(testX)
        rf_predict = clf2.predict(testX)

        tree_accuracy = getAccuracy(testY, treeY)
        rf_accuracy = getAccuracy(testY, rf_predict)

        tree_sum_accuracy += tree_accuracy
        rf_sum_accuracy += rf_accuracy

        trainX = []; trainY = []
        testX = []; testY = []

    tree_avg_accuracy = tree_sum_accuracy/N
    rf_avg_accuracy = rf_sum_accuracy/N

    avg_accuracy.append(tree_avg_accuracy)
    avg_accuracy.append(rf_avg_accuracy)

    avg_accuracy = np.array(avg_accuracy)
    return avg_accuracy

# dataX = getSynLinearDataSet(0.7, 10, 0.3)
# dataY = getLinearDataPrediction(0.7, dataX)
# print(dataY)

# observations = 1000
# print('• Number of Features: ',2)
# print('• Number of Observations: ', observations)
# linearDataAccuracy = getAccuracyLinearData(0.8, 0.4, observations, 0.2, 100)
# print('Tree accuracy: ', linearDataAccuracy[0])
# print('Rfor accuracy: ', linearDataAccuracy[1])
