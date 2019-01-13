from sgd_optimize import *
from classifiers import *
from data_generation import *
from single_run import *
from multiprocessing import Pool, Manager
from sklearn.model_selection import KFold
import sys
import csv
import gc
import os
from timeit import default_timer as timer

def cross_validateATIT(kernels, kernel_params, reg_params, X_train, y_train, 
                       quantile, loss_function, opt_type, opt_params):
    p = Pool()
    arg_list = []
    kf = KFold(n_splits=5)
    kf.get_n_splits(X_train)
    fold = 0
    for train_index, test_index in kf.split(X_train):
        fold += 1
        curr_X_train, curr_X_test = X_train[train_index], X_train[test_index]
        curr_y_train, curr_y_test = y_train[train_index], y_train[test_index]
        for kernel_type in kernels:
            curr_kernel_params = kernel_params[kernel_type]
            curr_reg_params = reg_params[kernel_type]
            for kernel_param in curr_kernel_params:
                for reg_param in curr_reg_params:
                    for surrogate in ['AT', 'IT']:
                        all_args = (surrogate, fold, kernel_param, reg_param, curr_X_train, curr_y_train, 
                                    curr_X_test, curr_y_test, quantile, loss_function, kernel_type, opt_type, opt_params)
                        arg_list.append(all_args)
    
    with open('cross_val_results.csv', 'w') as f:
        f.write('Fold,Surrogate,Loss_Function,Kernel_Type,Kernel_Parameter,Reg_Paremeter,01_Loss,' + 
                'Weighted_Loss,inSample01,inSampleWeighted,Time\n')
        writer = csv.writer(f, lineterminator = '\n', delimiter=",")
        for result in p.imap_unordered(single_run_star, arg_list):
            writer.writerow(result)

def cross_validate_kernel_grid(kernel_type, kernel_params, reg_params, X_train, y_train, quantiles, 
                              loss_function, opt_type, opt_params, cv_dir_name_base):
    p = Pool()
    manager = Manager()
    arg_list = []
    
    kf = KFold(n_splits=5)
    kf.get_n_splits(X_train)
    
    cv_file_name = cv_dir_name_base + '/results.csv'
    
    fold = 0
    for train_index, test_index in kf.split(X_train):
        fold += 1
        curr_X_train, curr_X_test = X_train[train_index], X_train[test_index]
        curr_y_train, curr_y_test = y_train[train_index], y_train[test_index]
        for quantile in quantiles:
            ## Make sure directory exists
            s = len(quantiles) + 1
            cv_dir_name = cv_dir_name_base + '/cv_results_i'+ str(int(round(quantile * s))) + 's' + str(s)
            if not os.path.exists(cv_dir_name):
                os.makedirs(cv_dir_name + '/loss')
                
            for surrogate in ['AT', 'IT']:
                for reg_param in reg_params:
                    for kernel_param in kernel_params:
                        all_args = (surrogate, fold, kernel_param, reg_param, curr_X_train, curr_y_train, curr_X_test, 
                                    curr_y_test, quantile, loss_function, kernel_type, opt_type, opt_params, cv_dir_name)
                        arg_list.append(all_args)
    
    ## Initialize results dictionary
    all_results = manager.dict()
    for quantile in quantiles:
        AT_dict = manager.dict()
        IT_dict = manager.dict()
        for r in reg_params:
            for k in kernel_params:
                AT_dict[str(r) + '_' + str(k)] = 0
                IT_dict[str(r) + '_' + str(k)] = 0
        
        all_results[quantile] = {'AT' : AT_dict, 'IT': IT_dict}
        
    print(all_results)
    with open(cv_file_name, 'w') as f:
        f.write('Fold,Surrogate,Quantile,Loss_Function,Kernel_Type,Kernel_Parameter,Reg_Paremeter,01_Loss,' + 
                'Weighted_Loss,inSample01,inSampleWeighted,Time\n')
        writer = csv.writer(f, lineterminator = '\n', delimiter=",")
        for result in p.imap_unordered(single_run_star, arg_list):
            ## Hardcoding from single_run
            kernel_param = result[5]
            reg_param = result[6]
            dict_string = str(reg_param) + '_' + str(kernel_param)
            
            abs_loss = float(result[8])
            quantile = result[2]
            
            AT_dict = all_results[quantile]['AT']
            IT_dict = all_results[quantile]['IT']
            
            if(result[1] == 'AT'):
                AT_dict[dict_string] += abs_loss
            elif(result[1] == 'IT'):
                IT_dict[dict_string] += abs_loss
            writer.writerow(result)
     
    best_params = {}
    AT_params = {}
    IT_params = {}
    for quantile in quantiles:
        AT_dict = all_results[quantile]['AT']
        IT_dict = all_results[quantile]['IT']
        
        AT_reg, AT_kern = min(AT_dict, key=AT_dict.get).split('_')
        IT_reg, IT_kern = min(IT_dict, key=IT_dict.get).split('_')
        
        AT_params[quantile] = {'reg': float(AT_reg), 'kernel': float(AT_kern)}
        IT_params[quantile] = {'reg': float(IT_reg), 'kernel': float(IT_kern)}
        
    best_params['AT'] = AT_params
    best_params['IT'] = IT_params
        
    p.close()
    return best_params


def get_params(best_params, surrogate):
    quantiles = []
    regs = []
    kernels = []
    
    curr_params = best_params[surrogate]
    
    for quantile in curr_params.keys():
        quantiles.append(quantile)
        regs.append(curr_params[quantile]['reg'])
        kernels.append(curr_params[quantile]['kernel'])
        
    return (quantiles, regs, kernels)
