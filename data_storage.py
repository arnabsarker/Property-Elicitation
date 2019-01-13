import numpy as np
from data_generation import *

def init(k):
    n = 100
    
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
    log('Retrieving data as matrices', style='w')
    train_set = np.genfromtxt(train_path, delimiter=',')
    test_set = np.genfromtxt(test_path, delimiter=',')

    X_train, y_train = get_features_and_labels(train_set, y_col)
    X_test, y_test = get_features_and_labels(test_set, y_col)

    #Normalize Data
    log('Normalizing data')
    X_train, y_train = (X_train - np.mean(X_train)), y_train
    X_test, y_test = (X_test - np.mean(X_test)) , y_test