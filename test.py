import numpy as np
import collections
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import bernoulli,poisson,norm,expon
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
# import graphviz
N = 20
# Slope
a = 0.7

dataX = []
X1 = []
X2 = []
for i in range(N):
    x1 = i/N
    X1.append(x1)

dataX.append(X1)

for i, x in enumerate(X1):
    error = np.random.uniform(-0.3,0.3)
    # print('error: ', error)
    x2 = a*x + error
    X2.append(x2)

dataX.append(X2)
dataX = np.transpose(np.array(dataX))
print(dataX)

# plt.plot(X1,X2, 'ro')
# plt.axis([0, 1, 0, 1])
# plt.show()

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

# ran = 4
# features = 2
# observations = 50
# num_simulations = 100
# accuracies = getRandNumAccuracy(ran, features, observations, num_simulations)
#
# print('• Number of Features: ',features)
# print('• Number of Observations: ', observations)
# print('• Tree Accuracy: ', accuracies[0])
# print('• Rfor Accuracy: ', accuracies[1])
