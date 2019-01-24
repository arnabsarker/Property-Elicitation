from classifiers import *
import numpy as np
import csv
from cross_validation import cross_validate_kernel_grid, get_params
from data_generation import get_features_and_labels, weighted_absolute_loss, zo_loss, generate_linear_complex_data
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

def log(string, style='a'):
    print(string)
    global dataset_name
    file_name = dataset_name + '_log.txt'
    with open(file_name, style) as file:
        file.write(str(string) + '\n')

if __name__ == '__main__':
    dataset_name = 'RCVRbf'
    
    train_path = 'datasets/rcv1_train5000.csv'
    test_path = 'datasets/rcv1_test5000.csv'
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
    
    '''
    log('Generating Data', style='w')
    k = 100
    n = 10000
    X_train, y_train = generate_linear_complex_data(k,n)
    X_test, y_test = generate_linear_complex_data(k,n)
    '''

    y_train = np.array(y_train).astype(int)
    y_test = np.array(y_test).astype(int)

    surrogate = 'AT'
    
    num_quantiles = int(round(5 * np.log10(k)))
    gammas= [1.0 * i / (num_quantiles + 1) for i in range(1, num_quantiles + 1)]
    log(gammas)

    loss_function = 'logistic'
    
    max_iter = 10 * X.shape[0]
    clf = NeuralNetQuantileMulticlass(surrogate=surrogate, gammas=gammas, max_iter=max_iter, loss_function=loss_function)

    clf.fit(X_train, y_train)

    preds_score = clf.predict_score(X_test)
    preds_vote = clf.predict(X_test)

    log('Absolute loss of score-based predictions')
    log(weighted_absolute_loss(preds_score, y_test, 0.5))
    log('Absolute loss of vote-based predictions')
    log(weighted_absolute_loss(preds_vote, y_test, 0.5))

    
    log('0-1 loss of score-based predictions')
    log(zo_loss(preds_score, y_test))
    log('0-1 loss of vote-based predictions')
    log(zo_loss(preds_vote, y_test))
    
    log('0-1 loss of Sklearn logistic regression')
    clf_log = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', fit_intercept=False).fit(X_train, y_train)
    log(zo_loss(clf_log.predict(X_test), y_test))
