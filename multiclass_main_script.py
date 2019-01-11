from classifiers import *
import numpy as np
from multiclass_results import get_features_and_labels
from data_generation import weighted_absolute_loss, zo_loss


if __name__ == '__main__':
    train_path = 'datasets/mnist_train5000.csv'
    test_path = 'datasets/mnist_test5000.csv'
    y_col = 0

    ## Import data and get sets
    print('Retrieving data as matrices')
    train_set = np.genfromtxt(train_path, delimiter=',')
    test_set = np.genfromtxt(test_path, delimiter=',')

    X_train, y_train = get_features_and_labels(train_set, y_col)
    X_test, y_test = get_features_and_labels(test_set, y_col)

    #Normalize Data
    print('Normalizing data')
    X_train, y_train = X_train - np.mean(X_train) , y_train
    X_test, y_test = X_test - np.mean(X_test) , y_test

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    k = np.unique(y_train).size # Number of classes

    surrogate = 'AT'
    num_quantiles = 5
    gammas= [1.0 * i / (num_quantiles + 1) for i in range(1, num_quantiles + 1)]
    print(gammas)

    kernel_type = 'rbf'
    alphas = [100] * num_quantiles
    kernel_param = 10

    loss_function = 'hinge'

    opt_type = 'Momentum'

    opt_params = {'learning_rate': 1e-6, 'momentum_gamma': 0.9, 'batch_size': 500}
    clf = QuantileMulticlass(surrogate=surrogate, gammas=gammas, alphas=alphas, kernel_type=kernel_type, kernel_param=kernel_param, loss_function=loss_function, opt_type=opt_type, opt_params=opt_params)

    clf.fit(X_train, y_train)

    preds_score = clf.predict_score(X_test)
    preds_vote = clf.predict(X_test)

    print('Absolute loss of score-based predictions')
    print(weighted_absolute_loss(preds_score, y_test, 0.5))
    print('Absolute loss of vote-based predictions')
    print(weighted_absolute_loss(preds_vote, y_test, 0.5))

    
    print('0-1 loss of score-based predictions')
    print(zo_loss(preds_score, y_test))
    print('0-1 loss of vote-based predictions')
    print(zo_loss(preds_vote, y_test))
