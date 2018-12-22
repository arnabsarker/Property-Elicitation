from classifiers import *
from data_generation import *
import gc
from timeit import default_timer as timer

def run_scripts():
    k = 3
    a = 0.3

    algs = ['SciPy', 'SGD', 'Momentum', 'Nesterov_Momentum', 'AMSGrad']

    kernels = ['linear', 'poly', 'rbf']

    kernel_params = {'linear': 1, 'poly': 5, 'rbf': 1}

    loss_functions = ['hinge', 'logistic']

    n_vals = [1000, 3000, 5000]

    params = {'SciPy': {}, 'Momentum': {'learning_rate': 1, 'momentum_gamma': 0.9},
             'SGD': {'learning_rate': 1},
             'Nesterov_Momentum': {'learning_rate': 1, 'momentum_gamma': 0.9}, 
             'AMSGrad': {'learning_rate': 1, 'beta1': 0.9, 'beta2': 0.99}}

    opt_file = open('opt_results.csv', "w")
    opt_file.write('Algorithm,Surrogate,Loss_Function,Kernel_Type,Kernel_Parameter,01_Loss,Weighted_Loss,Time,n\n')

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

                    y_quantiles = compute_alpha_quantile(X_test, a).astype(int)
                    y_quantiles_in = compute_alpha_quantile(X_train, a).astype(int)

                    opt_params['plot_file'] = 'imgs/' + alg + 'IT' + kernel + loss_function + str(n) + '.png'
                    print('Generating ' + opt_params['plot_file'])
                    start = timer()
                    clf5 = QuantileIT(gamma=a, alpha=0.01, kernel_type=kernel, opt_type=opt_type, opt_params=opt_params,
                                          kernel_param=kernel_param, loss_function=loss_function)
                    clf5.fit(X_train, y_train)
                    preds_5 = clf5.predict(X_test)
                    end = timer()
                    abs_loss = weighted_absolute_loss(preds_5, y_test, a)
                    zo_loss = metrics.zero_one_loss(preds_5, y_quantiles)

                    result_string = alg + ',' + 'IT' + ',' + loss_function + ',' + kernel + ',' + str(kernel_param) + ',' + str(zo_loss) + ',' + str(abs_loss) + ',' + str(end - start) + ',' + str(n)
                    print(result_string)
                    opt_file.write(result_string + '\n')

                    opt_params['plot_file'] = 'imgs/' + alg + 'AT' + kernel + loss_function + str(n) + '.png'
                    print('Generating ' + opt_params['plot_file'])
                
                    start = timer()
                    clf6 = QuantileAT(gamma=a, alpha=0.01, kernel_type=kernel, opt_type=opt_type, opt_params=opt_params,
                                          kernel_param=kernel_param, loss_function=loss_function)
                    clf6.fit(X_train, y_train)
                    preds_6 = clf6.predict(X_test)
                    end = timer()
                    abs_loss = weighted_absolute_loss(preds_6, y_test, a)
                    zo_loss = metrics.zero_one_loss(preds_6, y_quantiles)

                    result_string = alg + ',' + 'AT' + ',' + loss_function + ',' + kernel + ',' + str(kernel_param) + ',' + str(zo_loss) + ',' + str(abs_loss) + ',' + str(end - start) + ',' + str(n)
                    print(result_string)
                    opt_file.write(result_string + '\n')

                    gc.collect()

    opt_file.close()
    return 1


if __name__ == '__main__':
    run_scripts()
        
