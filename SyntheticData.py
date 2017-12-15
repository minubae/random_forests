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

def getSynBernoulliDataset(features, observations, probability):
    p = features
    n = observations
    prob = probability
    train_x  = []
    for i in range(p):
        #x_i is a random variable for each feature
        # dice = np.random.randint(n)
        x_i = bernoulli.rvs(prob, size=n)
        train_x.append(x_i)

    train_x = np.transpose(np.array(train_x))
    return train_x
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
def prediction_rule(input_x, prob):

    x = []
    x = input_x
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
    for i, x_i in enumerate(X):
        # print('x: ', x)
        # y = logical_or(x)
        y = prediction_rule(x_i, 0.5)
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

def getSynLinearDataset(features,observations, error):

    data_set = []
    p = features
    n = observations
    err = error

    #np.random.rand(d0,d1,...,dn):
    #Create an array of the given shape and populate it with random samples
    #from a uniform distribution over[0,1).
    beta = np.random.uniform(-1, 1, p-1)#np.random.rand(p)
    # beta = np.random.normal(0, 1, p-1)

    # error_vec = np.random.uniform(-err, err, n)
    error_vec = np.random.normal(0, err, n)

    x_data = np.random.rand(n,p-1)
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
        # tree_accuracy = getAccuracy(test_y, tree_y)
        # rf_accuracy = getAccuracy(test_y, rf_predict)
        tree_accuracy = accuracy_score(test_y, tree_y)
        rf_accuracy = accuracy_score(test_y, rf_predict)

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

def getDataVisualization(data_type, features, observastions, error):

    p = features
    n = observastions
    err = error
    type_x = data_type

    if type_x == 'linear':

        data_x = getSynLinearDataset(p, n, err)
        data_x_trp = np.transpose(data_x)

        if p == 2:
            plt.title('Observations (n)= %d' % n)
            plt.plot(data_x_trp[0], data_x_trp[1], 'ro', label='Observed Data')
            plt.plot(data_x_trp[2], label='Decision Rule')
            plt.xlabel('X1', fontsize=12)
            plt.ylabel('X2', fontsize=12)
            plt.axis([0, 1, -1, 1])
            # plt.axis([-0.5, 1.5, -0.5, 1.5])
            plt.legend(loc='upper right')

            plt.savefig('Data_%d.png' % n)

        elif p == 3:
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

    elif type_x == 'bernoulli':

        data_x = getSynBernoulliDataset(p, n, 0.5)
        data_x_trp = np.transpose(data_x)

        if p == 2:

            plt.title('Observations (n)= %d' % n)
            plt.plot(data_x_trp[0], data_x_trp[1], 'ro', label='Observed Data')
            # plt.plot(data_x_trp[2], label='Decision Rule')
            plt.xlabel('X1', fontsize=12)
            plt.ylabel('X2', fontsize=12)
            # plt.axis([0, 1, -1, 1])
            plt.axis([-0.5, 1.5, -0.5, 1.5])
            plt.legend(loc='upper right')

            plt.savefig('Data_%d.png' % n)

        elif p == 3:

            fig = plt.figure(figsize=(7,5))
            ax = Axes3D(fig)

            # plot data
            line = ax.plot(data_x_trp[0],data_x_trp[1],data_x_trp[2],'ok')
            ax.set_title('Observations (n)= %d' % n)
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
        # if data_type == 'linear':
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
max_n = 100
n_int = 10

err = 0.2
rf_state = 0
rf_depth = 10
simulations = 100

features = 2
observations = 100
# Data Type: 'bernoulli', 'linear'
data_type = 'linear'
getDataVisualization(data_type, features, observations, err)
# getComparisonVisualization(min_p, max_p, min_n, max_n, p_int, n_int, err, simulations, rf_depth, rf_state)
