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

# Creating Synthetic Bernoulli Dataset
def getBernoulliDataset(features, observations, probability):
    p = features
    n = observations
    prob = probability
    x_i = []; train_x  = []

    for i in range(n):
        x_i = np.random.binomial(1, prob, p)
        train_x.append(x_i)

    train_x = np.array(train_x)

    return train_x

# Creating Synthetic Linear Dataset
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
    # beta = np.random.uniform(0.5, 1, p-1)

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

# Creating Synthetic Independently Normal Distributed Dataset
def getSynIndNormalDataset(features, observations, mu, variance):

    p = features
    n = observations
    data_x = []

    m = mu
    var = variance

    for i in range(n):
        x_i = np.random.normal(m, var, p)
        # x_i = np.random.uniform(-1, 1, p)
        data_x.append(x_i)

    data_x = np.array(data_x)

    return data_x

# Computing Accuracy of Predictions
def getAccuracy(prediction_1, prediction_2):

    result = 0
    coincidence = 0
    test_y = prediction_1
    prediction = prediction_2

    n = len(test_y)

    for i in range(n):
        if(test_y[i] == prediction[i]):
            coincidence += 1

    result = coincidence/n
    return result

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
def logical_count(input_x, prob):

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

# Creating Prediction Rules for Y
def getBernoulliPrediction(data):
    X = data
    Y =[]
    for i, x_i in enumerate(X):
        # print('x: ', x)
        y = logical_or(x_i)
        # y = logical_count(x_i, 0.5)
        # y = logical_and(x_i)
        # print('y (logical_or): ', y)
        # print('y (logical_and): ', y2)
        Y.append(y)
        print(y)

    Y = np.array(Y)
    return Y

def getLinearDataPrediction(data):

    prediction = []
    data_x = data
    n, p = data_x.shape

    for x in data_x:

        if x[p-2] <= x[p-1]:
            y = 1
            prediction.append(y)
            print(y)
        else:
            y = 0
            prediction.append(y)
            print(y)

    prediction = np.array(prediction)
    return prediction

def getNormalDataPrediction(data, features, portion, shuffle):

    temp = []
    output = []
    predic_vec = []
    result = 0

    random = shuffle

    jump = portion
    data_x = data
    p = features

    criter_vec = np.random.normal(0, 0.2, p)
    # criter_vec = np.random.uniform(-1, 1, p)
    coins = np.random.binomial(1, 0.5, p)

    # print('coins:', coins)
    # print('criterion : ', criter_vec)

    for x in data_x:

        p = len(x)

        if random == "shuffle":
            np.random.shuffle(x)
            # print('shuffled_x: ', x)

        for i in range(0, p, jump):

            if x[i] >= criter_vec[i]:
                result = 1
                temp.append(result)
            else:
                result = 0
                temp.append(result)

            # if coins[i] == 0:
            #
            #     if x[i] >= criter_vec[i]:
            #         result = 1
            #         temp.append(result)
            #     else:
            #         result = 0
            #         temp.append(result)
            # else:
            #
            #     if x[i] <= criter_vec[i]:
            #         result = 1
            #         temp.append(result)
            #     else:
            #         result = 0
            #         temp.append(result)

        predic_vec.append(temp)
        temp = []

    predic_vec = np.array(predic_vec)

    # print('predic: ')
    # print(predic_vec)

    for y in predic_vec:

        num = logical_count(y, 0.5)
        output.append(num)
        print(num)

        # y_i = np.sum(y)
        # output.append(y_i)
        # print(y_i)

    output = np.array(output)

    return output

# Getting Accuracies from Decision Tree and Random Forest Predictions #rf_depth, rf_state
def getAccuracyPredictions(data_type, features, observations, error, num_simulations, mu, variance, portion, shuffle):

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
    type_x = data_type
    N = num_simulations
    n = observations
    p = features

    m = mu
    var = variance

    depth = rf_depth
    state = rf_state

    por = portion
    random = shuffle

    for i in range(N):

        if type_x == 'linear':

            train_x = getSynLinearDataset(p, n, err)
            train_y = getLinearDataPrediction(train_x)

            test_x = getSynLinearDataset(p, n, err)
            test_y = getLinearDataPrediction(test_x)

        elif type_x == 'bernoulli':

            train_x = getBernoulliDataset(p, n, 0.5)
            train_y = getBernoulliPrediction(train_x)

            test_x = getBernoulliDataset(p, n, 0.5)
            test_y = getBernoulliPrediction(test_x)

        elif type_x == 'normal':

            train_x = getSynIndNormalDataset(p, n, m, var)
            train_y = getNormalDataPrediction(train_x, p, por, random)

            test_x = getSynIndNormalDataset(p, n, m, var)
            test_y = getNormalDataPrediction(test_x, p, por, random)


        clf1 = tree.DecisionTreeClassifier()
        clf1 = clf1.fit(train_x,train_y)

        clf2 = RandomForestClassifier() #max_depth=depth, random_state=state
        clf2 = clf2.fit(train_x, train_y)

        tree_y = clf1.predict(test_x)
        rf_predict = clf2.predict(test_x)

        # tree_accuracy = accuracy_score(test_y, tree_y)
        # rf_accuracy = accuracy_score(test_y, rf_predict)
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

def getSumSquaresCovariance(data):
    x = data
    cov_x = np.cov(x)
    sum_squares = 0
    temp = []

    for i, xi in enumerate(cov_x):
        for j, x in enumerate(xi):
            if i<j:
                temp.append(x**2)

    temp = np.array(temp)
    sum_squares = np.sum(temp)

    return sum_squares

# Getting Data Visualization of a Synthetic Dataset
def getDataVisualization(data_type, features, observastions, error, mu, variance):

    p = features
    n = observastions
    err = error
    type_x = data_type

    m = mu
    var = variance

    if type_x == 'linear':

        data_x = getSynLinearDataset(p, n, err)
        data_x_trp = np.transpose(data_x)

        if p == 2:
            plt.title('Observations (n)= %d' % n)
            plt.suptitle('Linearly Separable Dataset', x=0.514, y=0.96, fontsize=10)
            plt.plot(data_x_trp[0], data_x_trp[1], 'ro', label='Observed Data')
            plt.plot(data_x_trp[2], label='Decision Rule')
            plt.xlabel('X1', fontsize=12)
            plt.ylabel('X2', fontsize=12)
            plt.axis([0, 1, -1, 1])
            # plt.axis([-0.5, 1.5, -0.5, 1.5])
            plt.legend(loc='upper right')

            plt.savefig('linear_data_%d.png' % n)

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

        data_x = getBernoulliDataset(p, n, 0.5)
        data_x_trp = np.transpose(data_x)

        if p == 2:

            plt.title('Observations (n)= %d' % n)
            plt.suptitle('Bernoulli Dataset (prob = 0.5)', x=0.514, y=0.96, fontsize=10)
            plt.plot(data_x_trp[0], data_x_trp[1], 'ro', label='Observed Data')
            # plt.plot(data_x_trp[2], label='Decision Rule')
            plt.xlabel('X1', fontsize=12)
            plt.ylabel('X2', fontsize=12)
            # plt.axis([0, 1, -1, 1])
            plt.axis([-0.5, 1.5, -0.5, 1.5])
            plt.legend(loc='upper right')

            plt.savefig('bernoulli_data_%d.png' % n)

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

    elif type_x == 'normal':

        data_x = getSynIndNormalDataset(p, n, m, var)
        data_x_trp = np.transpose(data_x)

        if p == 2:

            plt.title('Observations (n) = %d' % n)
            plt.suptitle('Normally Distributed Data ($\mu = 0, \sigma^2 = %.1f$)' %var, x=0.514, y=0.96, fontsize=10)
            # plt.text(1.5, 2.3, r'$\mu = %d$' %m)
            # plt.text(1.5, 2.1, r'$\sigma^2 = %.1f$' % var)

            # print('variance: ', var)
            plt.plot(data_x_trp[0], data_x_trp[1], 'ro', label='Observed Data')
            # plt.plot(data_x_trp[2], label='Decision Rule')
            plt.xlabel('X1', fontsize=12)
            plt.ylabel('X2', fontsize=12)
            # plt.axis([0, 1, -1, 1])
            plt.axis([-4, 4, -4, 4])
            plt.legend(loc='upper right')

            plt.savefig('normal_data_%.1f.png' % var)

        elif p == 3:

            fig = plt.figure(figsize=(7,5))
            ax = Axes3D(fig)

            # plot data
            line = ax.plot(data_x_trp[0],data_x_trp[1],data_x_trp[2],'ok')
            ax.set_title('Observations (n)= %d' % n)
            #modify axes
            ax.set_xlim(-2, 2)
            ax.set_ylim(2, -2)
            ax.minorticks_on()
            ax.tick_params(axis='both',which='minor',length=5,width=2,labelsize=10)
            ax.tick_params(axis='both',which='major',length=8,width=2,labelsize=10)

        #display
        plt.show()

    else:

        return "Sorry, We can not visualize your data."

# Getting Data Visualization of Accuracies from Decision Tree and Random Forest Predictions # rf_depth, rf_state
def getComparisonVisualization(data_type, features, observations, p_interval, n_interval, error, num_simulations, mu, variance, shuffle):

    p = features
    n = observations
    p_int = p_interval
    n_int = n_interval
    err = error
    simulations = num_simulations
    depth = rf_depth
    state = rf_state
    type_x = data_type

    comparison = []
    feature_vec = []
    acc_vec = []
    temp = []

    m = mu
    var = variance
    random = shuffle

    # try:

    for i in range(2, p, p_int):
        feature_vec.append(i)
        acc_vec = getAccuracyPredictions(type_x, i, n, err, simulations, m, var, p_int, random)
        comparison.append(acc_vec)

    comparison = np.array(comparison)
    # comparison = np.insert(comparison, 0, feature_vec, axis=1)
    comparison = np.transpose(comparison)


        # fig, axes = plt.subplots(nrows=5, ncols=2)
        # ax0, ax1, ax2, ax3, ax4, ax5 = axes.flatten()
        # index = 0
        # for n in range(min_n, max_n, n_int):
        # comparison = getComparison(type_x, p, n, p_int, err, simulations, depth, state)

        # features = comparison[0]
    tree_accuracy = comparison[0]
    rf_accuracy = comparison[1]


    if type_x == 'bernoulli':

        plt.title('Observations (n)= %d' % n)
        plt.suptitle('Bernoulli Data (prob = 0.5)', x=0.514, y=0.96, fontsize=10)
        plt.plot(feature_vec, tree_accuracy, '-o', label='Decision Tree')
        plt.plot(feature_vec, rf_accuracy, '-o', label='Random Forests')
        plt.xlabel('number of features (p)', fontsize=12)
        plt.ylabel('prediction accuracies (%)', fontsize=12)
        plt.legend(loc='lower right')

        plt.savefig('bernoulli_figure_%d.png' % n)

    elif type_x == 'linear':

        plt.title('Observations (n)= %d' % n)
        plt.suptitle('Linearly Separable Data', x=0.514, y=0.96, fontsize=10)
        plt.plot(feature_vec, tree_accuracy, '-o', label='Decision Tree')
        plt.plot(feature_vec, rf_accuracy, '-o', label='Random Forests')
        plt.xlabel('number of features (p)', fontsize=12)
        plt.ylabel('prediction accuracies (%)', fontsize=12)
        plt.legend(loc='upper right')

        plt.savefig('linear_figure_%d.png' % n)


    elif type_x == 'normal':

        plt.title('Observations (n) = %d' % n)
        plt.suptitle('Normally Distributed Data ($\mu = 0, \sigma^2 = %.1f$)' %var, x=0.514, y=0.96, fontsize=10)
        plt.text(0, 0, r'$\mu = %d$' %m)
        # plt.text(0, 0, r'$\sigma^2 = %.1f$' % var)

        plt.plot(feature_vec, tree_accuracy, '-o', label='Decision Tree')
        plt.plot(feature_vec, rf_accuracy, '-o', label='Random Forests')
        plt.xlabel('number of features (p)', fontsize=12)
        plt.ylabel('prediction accuracies (%)', fontsize=12)
        plt.legend(loc='upper right')

        plt.savefig('normal_figure_%.1f.png' % var)

    plt.show()

        # return comparison

    # except ValueError:
    #
    #     print('Number of features should be greater than 2. Try again.')


rf_state = 0
rf_depth = 10

err = 0.2
p_int = 2
n_int = 10

mu = 0
variance = 0.2
# variance = 0.8
# variance = 1.6
# variance = 3.2

simulations = 100

features = 2
observations = 1000

# features = 100
# observations = 200
# observations = 600
# observations = 800
# observations = 1000
random = 'shuffle'
# random = 'nope'

# Data Type: 'bernoulli', 'linear', 'normal'
# data_type = 'bernoulli'
# data_type = 'normal'
data_type = 'linear'


# data = getSynIndNormalDataset(10, 10, mu, variance)
# print(data)
# print('prediction')
# print(getNormalDataPrediction(data, 10, 1, 'shuffle'))
getDataVisualization(data_type, features, observations, err, mu, variance)
# getComparisonVisualization(data_type, features, observations, p_int, n_int, err, simulations, mu, variance, random)
