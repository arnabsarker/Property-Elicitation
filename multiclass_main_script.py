from classifiers import *
import numpy as np
from data_generation import get_features_and_labels, weighted_absolute_loss, zo_loss

def log(string):
    print(string)
    file_name = 'MNIST500_log.txt'
    with open(file_name, 'a') as file:
        file.write(string)

if __name__ == '__main__':
    dataset_name = 'MNIST500'
    train_path = 'datasets/mnist_train500.csv'
    test_path = 'datasets/mnist_test500.csv'
    y_col = 0

    ## Import data and get sets
    log('Retrieving data as matrices')
    train_set = np.genfromtxt(train_path, delimiter=',')
    test_set = np.genfromtxt(test_path, delimiter=',')

    X_train, y_train = get_features_and_labels(train_set, y_col)
    X_test, y_test = get_features_and_labels(test_set, y_col)

    #Normalize Data
    log('Normalizing data')
    X_train, y_train = X_train - np.mean(X_train) , y_train
    X_test, y_test = X_test - np.mean(X_test) , y_test

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    k = np.unique(y_train).size # Number of classes

    surrogate = 'AT'
    
    num_quantiles = 5
    gammas= [1.0 * i / (num_quantiles + 1) for i in range(1, num_quantiles + 1)]
    log(gammas)

    kernel_type = 'rbf'
    loss_function = 'hinge'
    
    opt_type = 'Momentum'
    opt_params = {'learning_rate': 1e-6, 'momentum_gamma': 0.9, 'batch_size': 100}
    
    
    log('Cross Validating parameters')
    cv_dir_name_base = 'cv_' + dataset_name
    kernel_params = [0.001, 0.1, 10, 1000]
    reg_params = [0.1, 10, 1000, 100000]
    
    best_parameters = cross_validate_kernel_grid(kernel_type, kernel_params, reg_params, X_train, y_train, gammas, 
                              loss_function, opt_type, opt_params, cv_dir_name_base)
    
    log(best_parameters)
    with open(dataset_name + '_cv_parameter_dict.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in best_parameters.items():
            writer.writerow([key, value])
            
    (gammas, alphas, kernel_params) = get_params(best_params, surrogate)
    
    log(gammas)
    log(alphas)
    log(kernel_params)
    
    clf = QuantileMulticlass(surrogate=surrogate, gammas=gammas, alphas=alphas, kernel_type=kernel_type, kernel_params=kernel_params, loss_function=loss_function, opt_type=opt_type, opt_params=opt_params)

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
