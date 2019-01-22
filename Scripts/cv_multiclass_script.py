from cross_validation import cross_validate_kernel_grid
import numpy as np
import os
import csv

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
    
    s = 6
    
    kernel_type = 'rbf'
    reg_params = [10, 100, 1000]
    kernel_params = [1, 10, 100, 1000]
    
    loss_function = 'hinge'
    
    opt_type = 'Momentum'
    
    opt_params = {'learning_rate': 5e-7, 'momentum_gamma': 0.9, 'batch_size': 100}
    
    cv_dir_name_base = 'cv_mnist_rbf2'
    
    quantiles = []
    cv_file_names = {}
    cv_dir_names = {}
    for i in range(1, s):
        quantile = i * 1.0 / s
        quantiles.append(quantile)
            
    print(quantiles)    
    best_parameters = cross_validate_kernel_grid(kernel_type, kernel_params, reg_params, X_train, y_train, quantiles, 
                              loss_function, opt_type, opt_params, cv_dir_name_base)
    
    print(best_parameters)
    with open('rbf_dict.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in best_parameters.items():
            writer.writerow([key, value])
