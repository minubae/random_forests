import numpy as np
import collections
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import bernoulli,poisson,norm,expon
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
# import graphviz

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

def getSynLinearDataset(features,observations, error):

    data_set = []
    p = features
    n = observations
    err = error

    #np.random.rand(d0,d1,...,dn):
    #Create an array of the given shape and populate it with random samples
    #from a uniform distribution over[0,1).
    beta = beta = np.random.uniform(-1, 1, p-1)#np.random.rand(p)
    error_vec = np.random.uniform(-err, err, n)

    x_data = np.random.rand(n,p-1)
    # x_data = np.random.rand(n,p-1)
    # x_data = np.insert(x_data, 0, 1, axis=1)

    x_new = np.dot(x_data, beta)
    x_new_err = np.add(x_new, error_vec)

    x_data = np.insert(x_data, p-1, x_new_err, axis=1)
    x_data = np.insert(x_data, p, x_new, axis=1)
    # x_data = np.delete(x_data, 0, 1)

    # data_set.append(x_new)
    # data_set.append(x_new_err)
    # data_set = np.transpose(np.array(data_set))
    return x_data

def getLinearDataPrediction(Data):

    prediction = []
    data_x = Data
    n, p = data_x.shape

    for x in data_x:

        if x[p-2] <= x[p-1]:
            y = 1
            prediction.append(y)
        else:
            y = 0
            prediction.append(y)

    prediction = np.array(prediction)
    return prediction

def getAccuracyLinearData(features, observations, error, num_simulations, rf_depth, rf_state):

    tree_accuracy = 0
    rf_accuracy = 0

    tree_sum_accuracy = 0
    rf_sum_accuracy = 0

    tree_avg_accuracy = 0
    rf_avg_accuracy = 0

    avg_accuracy = []
    train_x = []; train_y = []
    test_x = []; test_y = []

    err = error

    N = num_simulations
    n = observations
    p = features

    depth = rf_depth
    state = rf_state

    for i in range(N):

        train_x = getSynLinearDataset(p, n, err)
        train_y = getLinearDataPrediction(train_x)

        test_x = getSynLinearDataset(p, n, err)
        test_y = getLinearDataPrediction(test_x)

        clf1 = tree.DecisionTreeClassifier()
        clf1 = clf1.fit(train_x,train_y)

        clf2 = RandomForestClassifier() #max_depth=depth, random_state=state
        clf2 = clf2.fit(train_x, train_y)

        tree_y = clf1.predict(test_x)
        rf_predict = clf2.predict(test_x)

        # print('trainprediction:', test_y)
        # print('tree_prediction:', tree_y)
        # print('rfor_predection:', rf_predict)
        # print('\n')
        tree_accuracy = getAccuracy(test_y, tree_y)
        rf_accuracy = getAccuracy(test_y, rf_predict)

        tree_sum_accuracy += tree_accuracy
        rf_sum_accuracy += rf_accuracy

        train_x = []; train_y = []
        test_x = []; test_y = []

    tree_avg_accuracy = tree_sum_accuracy/N
    rf_avg_accuracy = rf_sum_accuracy/N

    avg_accuracy.append(tree_avg_accuracy)
    avg_accuracy.append(rf_avg_accuracy)

    avg_accuracy = np.array(avg_accuracy)
    return avg_accuracy

def getDataVisualization(Data, Features, Observastions):

    features = Features
    observations = Observastions
    data_x = Data

    data_x_trp = np.transpose(data_x)

    if features == 2:

        plt.plot(data_x_trp[0], data_x_trp[1], 'ro')
        plt.plot(data_x_trp[2])
        plt.axis([-0.5, 1.5, -0.5, 1.5])
        plt.show()

    elif features == 3:

        fig = plt.figure(figsize=(7,5))
        ax = Axes3D(fig)

        # plot data
        line1 = ax.plot(data_x_trp[0],data_x_trp[1],data_x_trp[2],'ok')
        line2 = ax.plot(x_data[0], x_data[1], x_data[3])
        # plt.plot(data_x_trp[2])
        #modify axes
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(1.5, -0.5)
        ax.minorticks_on()
        ax.tick_params(axis='both',which='minor',length=5,width=2,labelsize=10)
        ax.tick_params(axis='both',which='major',length=8,width=2,labelsize=10)

        #display
        plt.show()

    else:

        return "Sorry, We can not visualize your data."

def getComparison(min_features, max_features, interval, observations, error, num_simulations, rf_depth, rf_state):

    err = error
    n = observations
    min_p = min_features
    max_p = max_features
    jump = interval
    simulations = num_simulations
    depth = rf_depth
    state = rf_state

    comparison = []
    features = []
    acc_vec = []
    temp = []

    for i in range(min_p, max_p, jump):
        features.append(i)
        acc_vec = getAccuracyLinearData(i, n, err, simulations, depth, state)
        comparison.append(acc_vec)

    comparison = np.array(comparison)
    comparison = np.insert(comparison, 0, features, axis=1)
    comparison = np.transpose(comparison)

    return comparison

def getComparisonVisualization(min_features, max_features, min_obs, max_obs, p_interval, n_interval, error, num_simulations, rf_depth, rf_state):

    min_p = min_features
    max_p = max_features
    min_n = min_obs
    max_n = max_obs
    p_int = p_interval
    n_int = n_interval
    err = error
    simulations = num_simulations
    depth = rf_depth
    state = rf_state


    # fig, axes = plt.subplots(nrows=5, ncols=2)
    # ax0, ax1, ax2, ax3, ax4, ax5 = axes.flatten()
    # index = 0
    # for n in range(min_n, max_n, n_int):
    comparison = getComparison(min_p, max_p, p_int, max_n, err, simulations, depth, state)

    features = comparison[0]
    tree_accuracy = comparison[1]
    rf_accuracy = comparison[2]

    plt.title('Observations (n)= %d' % max_n)
    plt.plot(features, tree_accuracy, '-o', label='Decision Tree')
    plt.plot(features, rf_accuracy, '-o', label='Random Forests')
    plt.xlabel('number of features (p)', fontsize=12)
    plt.ylabel('prediction accuracies (%)', fontsize=12)
    plt.legend(loc='upper right')

        # index += 1


    # fig.tight_layout()
    plt.savefig('Figure_%d.png' % max_n)
    plt.show()

    return comparison

min_p = 2
max_p = 100
p_int = 3

min_n = 10
max_n = 1000
n_int = 10

err = 0.3
rf_state = 0
rf_depth = 10
simulations = 100

getComparisonVisualization(min_p, max_p, min_n, max_n, p_int, n_int, err, simulations, rf_depth, rf_state)



# data = getSynLinearDataset(p, n, err)
# print(data)
# print(getLinearDataPrediction(data))
# print(getAccuracyLinearData(p,n,err,simulations,rf_depth,rf_state))
# getDataVisualization(data, p, n)
