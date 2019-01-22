import numpy as np
from data_generation import *

def init():
    global X_train
    global y_train
    global X_test
    global y_test
    
    train_path = 'datasets/sector_train.csv'
    test_path = 'datasets/sector_test.csv'
    y_col = 0
    k = 105
    classes = np.arange(k)
    
    
    ## Import data and get sets
    train_set = np.genfromtxt(train_path, delimiter=',')
    test_set = np.genfromtxt(test_path, delimiter=',')

    X_train, y_train = get_features_and_labels(train_set, y_col)
    X_test, y_test = get_features_and_labels(test_set, y_col)

    #Normalize Data
    X_train, y_train = (X_train - np.mean(X_train)), y_train
    X_test, y_test = (X_test - np.mean(X_test)) , y_test


    y_train = y_train.astype(int)
    y_test = y_test.astype(int)


def init_rcv1():
    global X_train
    global y_train
    global X_test
    global y_test

    train_path = 'datasets/rcv1_train5000.csv'
    test_path = 'datasets/rcv1_test5000.csv'
    y_col = 0
    k = 53
    classes = np.arange(k)


    ## Import data and get sets
    train_set = np.genfromtxt(train_path, delimiter=',')
    test_set = np.genfromtxt(test_path, delimiter=',')

    X_train, y_train = get_features_and_labels(train_set, y_col)
    X_test, y_test = get_features_and_labels(test_set, y_col)

    #Normalize Data
    X_train, y_train = (X_train - np.mean(X_train)), y_train
    X_test, y_test = (X_test - np.mean(X_test)) , y_test

    n = 1000
    train_indices = np.random.choice(5000, n, replace=False)
    test_indices = np.random.choice(5000, n, replace=False)
    X_train, y_train = X_train[train_indices, :], y_train[train_indices]
    X_test, y_test = X_test[test_indices, :], y_test[test_indices]

    y_train = np.array(y_train).astype(int)
    y_test = np.array(y_test).astype(int)
