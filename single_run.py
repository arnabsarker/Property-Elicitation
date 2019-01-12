from sgd_optimize import *
from classifiers import *
from data_generation import weighted_absolute_loss
from single_run import *
from multiprocessing import Pool, Manager
from sklearn.model_selection import KFold
import sys
import csv
import gc
from timeit import default_timer as timer

def single_run_star(one_arg):
    return single_run(*one_arg)

def single_run_opt_star(one_arg):
    return single_run_opt(*one_arg)
    
def single_run_opt(surrogate, kernel_param, reg_param, X_train, y_train, X_test, y_test,
               quantile, loss_function, kernel_type, opt_type, opt_params, opt_dir_name):
    a = quantile

    if(surrogate == 'IT'):
        name = opt_type + 'IT' + kernel_type + str(kernel_param) + loss_function + str(reg_param) + '.png'
        opt_params['plot_file'] = opt_dir_name + '/loss/' + name
        start = timer()
        clf = QuantileIT(gamma=a, alpha=reg_param, kernel_type=kernel_type, opt_type=opt_type, opt_params=opt_params,
                              kernel_param=kernel_param, loss_function=loss_function)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        end = timer()

    elif(surrogate == 'AT'): 
        name = opt_type + 'AT' + kernel_type + str(kernel_param) + loss_function + str(reg_param) + '.png'
        opt_params['plot_file'] = opt_dir_name + '/loss/' + name

        start = timer()
        clf = QuantileAT(gamma=a, alpha=reg_param, kernel_type=kernel_type, opt_type=opt_type, opt_params=opt_params,
                              kernel_param=kernel_param, loss_function=loss_function)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        end = timer()

    abs_loss = weighted_absolute_loss(preds, y_test, a)
    preds_in = clf.predict(X_train)
    abs_loss_in = weighted_absolute_loss(preds_in, y_train, a)

    gc.collect()
    return (opt_type, surrogate, quantile, loss_function, kernel_type, kernel_param, 
            reg_param, abs_loss, abs_loss_in, end - start)

def single_run(surrogate, fold, kernel_param, reg_param, X_train, y_train, X_test, y_test,
               quantile, loss_function, kernel_type, opt_type, opt_params, cv_dir_name):
    a = quantile
    y_quantiles = y_test #compute_alpha_quantile(X_test, a).astype(int) only for synthetic data
    y_quantiles_in = y_train #compute_alpha_quantile(X_train, a).astype(int)

    if(surrogate == 'IT'):
        name = 'Fold' + str(fold) + opt_type + 'IT' + kernel_type + str(kernel_param) + loss_function + str(reg_param) + '.png'
        opt_params['plot_file'] = cv_dir_name + '/loss/' + name
        start = timer()
        clf = QuantileIT(gamma=a, alpha=reg_param, kernel_type=kernel_type, opt_type=opt_type, opt_params=opt_params,
                              kernel_param=kernel_param, loss_function=loss_function)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        end = timer()

    elif(surrogate == 'AT'): 
        name = 'Fold' + str(fold) + opt_type + 'AT' + kernel_type + str(kernel_param) + loss_function + str(reg_param) + '.png'
        opt_params['plot_file'] = cv_dir_name + '/loss/' + name

        start = timer()
        clf = QuantileAT(gamma=a, alpha=reg_param, kernel_type=kernel_type, opt_type=opt_type, opt_params=opt_params,
                              kernel_param=kernel_param, loss_function=loss_function)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        end = timer()

    abs_loss = weighted_absolute_loss(preds, y_test, a)
    zo_loss = metrics.zero_one_loss(preds, y_quantiles)
    preds_in = clf.predict(X_train)
    abs_loss_in = weighted_absolute_loss(preds_in, y_train, a)
    zo_loss_in = metrics.zero_one_loss(preds_in, y_quantiles_in)

    '''
    # Plot decision boundary
    plt.figure(0)
    fig=plt.figure(figsize=(6,6))
    plt.scatter([X_test[:, 0]], [X_test[:, 1]], c=[preds])
    plt.savefig(cv_dir_name + '/boundaries/' + name)
    plt.close()
    '''
    print(abs_loss)
    gc.collect()
    return (fold, surrogate, quantile, loss_function, kernel_type, kernel_param, 
            reg_param, zo_loss, abs_loss, zo_loss_in, abs_loss_in, end - start)
