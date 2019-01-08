from cross_validation import cross_validate_linear_reg
import numpy as np
import os
import csv

def get_features_and_labels(dataset, y_col):
    y = dataset[:, y_col:y_col+1]
    selector = [col for col in range(dataset.shape[1]) if col != y_col]
    X = dataset[:, selector]
    return X, y

if __name__ == '__main__':
    train_path = 'datasets/mnist_train.csv'
    test_path = 'datasets/mnist_test.csv'
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
    
    k = np.unique(y_train).size # Number of quantiles
    
    s = 6
    
    reg_params = [1, 10, 100, 1000]
    
    loss_function = 'hinge'
    
    opt_type = 'Momentum'
    
    opt_params = {'learning_rate': 1e-8, 'batch_size': 500}
    
    cv_file_name = 'cv_mnist/results.csv'
    
    quantiles = []
    cv_file_names = {}
    cv_dir_names = {}
    for i in range(1, s):
        quantile = i / s
        
        cv_dir_name = 'cv_mnist/cv_results_i' + str(i) + 's' + str(s) + '_imgs'
        cv_dir_names[quantile] = cv_dir_name
        
        if not os.path.exists(cv_dir_name):
            os.makedirs(cv_dir_name + '/loss')
            os.makedirs(cv_dir_name + '/boundaries')
        
        quantiles.append(quantile)
            
        
    best_parameters = cross_validate_linear_reg(reg_params, X_train, y_train, quantiles, 
                              loss_function, opt_type, opt_params, cv_file_name, cv_dir_names)
    
    print(best_parameters)
    with open('dict.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in best_parameters.items():
            writer.writerow([key, value])