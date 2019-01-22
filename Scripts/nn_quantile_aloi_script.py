import sys
sys.path.append("..")
import numpy as np
import nn_classifiers as nc
from data_generation import generate_simplex_data, get_features_and_labels, weighted_absolute_loss, zo_loss
from matplotlib import pyplot as plt

import pickle

def log(string, style='a'):
    print(string)
    global dataset_name
    file_name = dataset_name + '_log.txt'
    with open(file_name, style) as file:
        file.write(str(string) + '\n')

if __name__ == '__main__':
    dataset_name = 'Aloi'
    try:

        train_path = '../datasets/aloi_train.csv'
        test_path = '../datasets/aloi_test.csv'
        y_col = 0


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

        y_train = np.array(y_train).astype(int)
        y_test = np.array(y_test).astype(int)

        nn_clf = nc.NeuralNetQuantile(gamma=0.5, max_iter=90000, 
                     surrogate_type='AT', loss_function='hinge')

        nn_clf.fit(X_train, y_train, use_multiprocessing=False)

        preds = nn_clf.predict(X_test)

        log('Absolute loss of score-based predictions')
        log(weighted_absolute_loss(preds, y_test, 0.5))

        nn_clf.nn_.save('nn_aloi_clf.h5')
        
    except Exception as e:
        log(e)
    
