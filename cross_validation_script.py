import numpy as np
from sklearn import base, metrics
from sklearn.model_selection import KFold
import pandas as pd
from classifiers import *
from data_generation import *
import sys

def compute_alpha_quantile(X, alpha, k):
    n = X.shape[0]
    y = np.zeros((n, 1))
    w_alpha = alpha * 10 ## Simplex weighted to be 10 x 10
    for i in range(0, n):
        curr_vec = X[i,:]
        
        if(w_alpha < curr_vec[0]):
            y[i, 0] = 0
        elif(w_alpha  > curr_vec[-1]):
            y[i,0] = k - 1
        else:
            y[i,0] = np.argmax(curr_vec > w_alpha)
    return y

def run_cross_validation(X_train, y_train):
    ## Cross Validation for rbf and polynomial kernel
    quantiles = [0.3, 0.5, 0.75]

    reg_vals = [10**(-2), 10**(-1), 10**0, 10**1]
    q_vals = [2, 3, 4, 5]
    gamma_vals = [10**(-1), 10**0, 10**1, 10**2]

    cv_results = pd.DataFrame(columns=['Fold', 'Quantile', 'Surrogate', 
                                       'Kernel_Type', 'Kernel_Parameter', 'Regularization', '01_Loss'])

    kf = KFold(n_splits=5)
    kf.get_n_splits(X_train)
    fold = 0
    for train_index, test_index in kf.split(X_train):
        fold += 1
        curr_X_train, curr_X_test = X_train[train_index], X_train[test_index]
        curr_y_train, curr_y_test = y_train[train_index], y_train[test_index]
        for a in quantiles:
            print('Fold ' + str(fold) + ' Quantile ' + str(a))
            a_test_quantiles = compute_alpha_quantile(curr_X_test, a, 3)
            for reg in reg_vals:
                for q in q_vals:
                    clf1 = LogisticQuantileIT(gamma=a, alpha=1., kernel_type='poly', kernel_param=q)
                    clf1.fit(curr_X_train, curr_y_train)
                    lossIT = metrics.zero_one_loss(clf1.predict(curr_X_test), a_test_quantiles , normalize=False)
                    result = {'Fold': fold, 'Quantile': a, 'Surrogate': 'IT', 'Kernel_Type': 'poly', 
                              'Kernel_Parameter': q, 'Regularization': reg, '01_Loss': lossIT}
                    cv_results = cv_results.append(result, ignore_index=True)

                    clf2 = LogisticQuantileAT(gamma=a, alpha=1., kernel_type='poly', kernel_param=q)
                    clf2.fit(curr_X_train, curr_y_train)
                    lossAT = metrics.zero_one_loss(clf2.predict(curr_X_test), a_test_quantiles , normalize=False)
                    result = {'Fold': fold, 'Quantile': a, 'Surrogate': 'AT', 'Kernel_Type': 'poly', 
                              'Kernel_Parameter': q, 'Regularization': reg, '01_Loss': lossAT}
                    cv_results = cv_results.append(result, ignore_index=True)

                for g in gamma_vals:
                    clf1 = LogisticQuantileIT(gamma=a, alpha=1., kernel_type='rbf', kernel_param=g)
                    clf1.fit(curr_X_train, curr_y_train)
                    lossIT = metrics.zero_one_loss(clf1.predict(curr_X_test), a_test_quantiles , normalize=False)
                    result = {'Fold': fold, 'Quantile': a, 'Surrogate': 'IT', 'Kernel_Type': 'rbf', 
                              'Kernel_Parameter': g, 'Regularization': reg, '01_Loss': lossIT}
                    cv_results = cv_results.append(result, ignore_index=True)

                    clf2 = LogisticQuantileAT(gamma=a, alpha=1., kernel_type='rbf', kernel_param=g)
                    clf2.fit(curr_X_train, curr_y_train)
                    lossAT = metrics.zero_one_loss(clf2.predict(curr_X_test), a_test_quantiles , normalize=False)
                    result = {'Fold': fold, 'Quantile': a, 'Surrogate': 'AT', 'Kernel_Type': 'rbf', 
                              'Kernel_Parameter': g, 'Regularization': reg, '01_Loss': lossAT}
                    cv_results = cv_results.append(result, ignore_index=True)
    return cv_results


if __name__ == '__main__':
    n = int(sys.argv[1])
    X_train,y_train = generate_simplex_data(3,n)
    y_train = y_train.astype(int)
    np.savetxt('X_train.csv', X_train, delimiter=',')
    np.savetxt('y_train.csv', y_train, delimiter=',')
    
    cv_results = run_cross_validation(X_train, y_train)
    
    cv_results.to_csv('cv_results.csv', index=False)
    