from classifiers import *
from data_generation import *
import gc
from timeit import default_timer as timer
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
    run_scripts()
        
