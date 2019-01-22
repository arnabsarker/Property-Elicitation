from sgd_optimize import *
from classifiers import *
from single_run import single_run_opt_star
from data_generation import weighted_absolute_loss
from multiprocessing import Pool, Manager
import sys
import csv
import gc
import os
from timeit import default_timer as timer

def get_features_and_labels(dataset, y_col):
    y = dataset[:, y_col:y_col+1]
    selector = [col for col in range(dataset.shape[1]) if col != y_col]
    X = dataset[:, selector]
    return X, y

def run_scripts():
    k = 3
    a = 0.3

    algs = ['SciPy', 'SGD', 'Momentum']

    kernels = ['linear', 'poly', 'rbf']

    kernel_params = {'linear': 1, 'poly': 5, 'rbf': 1}

    loss_functions = ['hinge', 'logistic']

    n_vals = [1000]

    params = {'SciPy': {}, 'Momentum': {'learning_rate': 0.0001, 'momentum_gamma': 0.9, 'batch_size': 200},
             'SGD': {'learning_rate': 0.0001, 'batch_size': 100}}

    opt_file = open('opt_results.csv', "w")
    opt_file.write('Algorithm,Surrogate,Loss_Function,Kernel_Type,Kernel_Parameter,01_Loss,Weighted_Loss,inSample01,inSampleWeighted,Time,n\n')

    for n in n_vals:
        X_train,y_train = generate_simplex_data(k,n)
        y_train = y_train.astype(int)

        X_test,y_test = generate_simplex_data(k,n)
        y_test = y_test.astype(int)
        for kernel in kernels:
            kernel_param = kernel_params[kernel]
            for loss_function in loss_functions:
                for alg in algs:
                    opt_type = alg
                    opt_params = params[alg]
                    run_ITAT(X_train, y_train, X_test, y_test, a, n, kernel, kernel_param, loss_function,
                             opt_type, opt_params, opt_file)  

    opt_file.close()
    return 1


def run_ITAT(X_train, y_train, X_test, y_test, a, n, kernel, kernel_param, loss_function, opt_type, opt_params, opt_file):
    y_quantiles = compute_alpha_quantile(X_test, a).astype(int)
    y_quantiles_in = compute_alpha_quantile(X_train, a).astype(int)

    name = opt_type + 'IT' + kernel + loss_function + str(n) + '.png'
    opt_params['plot_file'] = 'imgs/loss/' + name
    start = timer()
    clf5 = QuantileIT(gamma=a, alpha=0.01, kernel_type=kernel, opt_type=opt_type, opt_params=opt_params,
                          kernel_param=kernel_param, loss_function=loss_function)
    clf5.fit(X_train, y_train)
    preds_5 = clf5.predict(X_test)
    end = timer()
    
    abs_loss = weighted_absolute_loss(preds_5, y_test, a)
    zo_loss = metrics.zero_one_loss(preds_5, y_quantiles)
    preds_5_in = clf5.predict(X_train)
    abs_loss_in = weighted_absolute_loss(preds_5_in, y_train, a)
    zo_loss_in = metrics.zero_one_loss(preds_5_in, y_quantiles_in)
    
    # Plot decision boundaries
    plt.figure(0)
    fig=plt.figure(figsize=(6,6))
    plt.scatter([X_test[:, 0]], [X_test[:, 1]], c=[preds_5])
    plt.savefig('imgs/boundaries/' + name)
    plt.close()


    result_string = opt_type + ',' + 'IT' + ',' + loss_function + ',' + kernel + ',' + str(kernel_param) + ',' + str(zo_loss) + ',' + str(abs_loss) +',' + str(zo_loss_in) + ',' + str(abs_loss_in) + ',' + str(end - start) + ',' + str(n)
    print(result_string)
    opt_file.write(result_string + '\n')

    name = opt_type + 'AT' + kernel + loss_function + str(n) + '.png'
    opt_params['plot_file'] = 'imgs/loss/' + name

    start = timer()
    clf6 = QuantileAT(gamma=a, alpha=0.01, kernel_type=kernel, opt_type=opt_type, opt_params=opt_params,
                          kernel_param=kernel_param, loss_function=loss_function)
    clf6.fit(X_train, y_train)
    preds_6 = clf6.predict(X_test)
    end = timer()
    
    abs_loss = weighted_absolute_loss(preds_6, y_test, a)
    zo_loss = metrics.zero_one_loss(preds_6, y_quantiles)
    preds_6_in = clf6.predict(X_train)
    abs_loss_in = weighted_absolute_loss(preds_6_in, y_train, a)
    zo_loss_in = metrics.zero_one_loss(preds_6_in, y_quantiles_in)

    # Plot decision boundary
    plt.figure(0)
    fig=plt.figure(figsize=(6,6))
    plt.scatter([X_test[:, 0]], [X_test[:, 1]], c=[preds_6])
    plt.savefig('imgs/boundaries/' + name)
    plt.close()


    result_string = opt_type + ',' + 'AT' + ',' + loss_function + ',' + kernel + ',' + str(kernel_param) + ',' + str(zo_loss) + ',' + str(abs_loss) + ',' + str(zo_loss_in) + ',' + str(abs_loss_in) + ',' + str(end - start) + ',' + str(n)
    print(result_string)
    opt_file.write(result_string + '\n')

    gc.collect()

def mnist_optimization():
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
    
    s = 4
    
    reg_param = 10
    
    loss_function = 'hinge'
    
    opt_types = ['SGD', 'Momentum', 'Nesterov_Momentum', 'AMSGrad']
    
    opt_params_dict = {'SGD': {'learning_rate': 1e-8, 'batch_size': 500},
                      'Momentum': {'learning_rate': 1e-8, 'momentum_gamma': 0.9, 'batch_size': 500},
                      'Nesterov_Momentum': {'learning_rate': 1e-8, 'momentum_gamma': 0.9, 'batch_size': 500},
                      'AMSGrad': {'learning_rate': 1e-8, 'beta1': 0.9, 'beta2': 0.99, 'batch_size': 500}}
    
    opt_file_name = 'mnist_opt/results.csv'
    
    quantiles = []
    opt_dir_names = {}
    for i in range(1, s):
        quantile = (1.0 * i) / s
        
        opt_dir_name = 'mnist_opt/results_i' + str(i) + 's' + str(s) + '_imgs'
        opt_dir_names[quantile] = opt_dir_name
        
        if not os.path.exists(opt_dir_name):
            os.makedirs(opt_dir_name + '/loss')
        
        quantiles.append(quantile)
            
        
    optimization_linear_reg(reg_param, X_train, y_train, X_test, y_test, quantiles, 
                            loss_function, opt_types, opt_params_dict, opt_file_name, opt_dir_names)

def optimization_linear_reg(reg_param, X_train, y_train, X_test, y_test, quantiles, 
                            loss_function, opt_types, opt_params_dict, opt_file_name, opt_dir_names):
    p = Pool()
    manager = Manager()
    arg_list = []
    
    kernel_type = 'linear'
    kernel_param = 1 ## Number is not relevant
    
    
    for quantile in quantiles:
        opt_dir_name = opt_dir_names[quantile]
        for opt_type in opt_types:
            opt_params = opt_params_dict[opt_type]
            for surrogate in ['AT', 'IT']:
                all_args = (surrogate, kernel_param, reg_param, X_train, y_train, X_test, y_test,
                            quantile, loss_function, kernel_type, opt_type, opt_params, opt_dir_name)
                arg_list.append(all_args)

    with open(opt_file_name, 'w') as f:
        f.write('Fold,Surrogate,Quantile,Loss_Function,Kernel_Type,Kernel_Parameter,Reg_Parameter,' + 
                'Weighted_Loss,inSampleWeighted,Time\n')
        writer = csv.writer(f, lineterminator = '\n', delimiter=",")
        for result in p.imap_unordered(single_run_opt_star, arg_list):
            writer.writerow(result)
    

if __name__ == '__main__':
    mnist_optimization()
        
