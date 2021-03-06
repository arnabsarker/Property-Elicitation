import numpy as np
import matplotlib.pyplot as plt

def optimize_batch(obj_function=None, x0=None, method='SGD', gradient=None, 
                   opt_params={},
                   bounds=None, epsilon=1e-8, max_iters=10000, min_iters=500, args=None):
    x = x0
    best_x = x0
    best_loss = 1e8
    
    batch_size = opt_params['batch_size']
    
    num_iters = 0
    loss_difference = 1e8
    new_loss = 1e8
    
    plot = 'plot_file' in opt_params
    if(plot):
        losses = []
    
    if(method == 'SGD'):
        learning_rate = opt_params['learning_rate']
    elif(method == 'Momentum' or method == 'Nesterov_Momentum'):
        learning_rate = opt_params['learning_rate']
        momentum_gamma = opt_params['momentum_gamma']
        v = 0
    elif(method == 'AMSGrad'):
        learning_rate = opt_params['learning_rate']
        beta1 = opt_params['beta1']
        beta2 = opt_params['beta2']
        eps = 1e-8
        v = 0
        v_hat = 0
        m = 0
    while(num_iters < max_iters and (loss_difference > epsilon or num_iters < min_iters)):
        old_loss = new_loss
        
        
        if(method == 'SGD'):
            eta = learning_rate
            x = x - eta * evaluate_gradient(gradient, batch_size, x, args)
        elif(method == 'Momentum'):
            grad = evaluate_gradient(gradient, batch_size, x, args)
            eta = learning_rate
            v = momentum_gamma * v + eta * grad
            x = x - v
        elif(method == 'Nesterov_Momentum'):
            weighted_v = momentum_gamma * v
            grad = evaluate_gradient(gradient, batch_size, x - weighted_v, args)
            eta = learning_rate
            v = weighted_v + eta * grad
            x = x - v
        elif(method == 'AMSGrad'):
            grad = evaluate_gradient(gradient, batch_size, x, args)
            eta = learning_rate
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            v_hat = np.maximum(v_hat, v)
            x = x - (eta / (np.sqrt(v_hat) + eps) ) * m
            
        new_loss = evaluate_loss(obj_function, batch_size, x, args)
        
        if(plot):
            losses.append(new_loss)
        
        loss_difference = np.abs(new_loss - old_loss) / old_loss
        
        num_iters += 1
        
        if(new_loss < best_loss):
            best_x = x
            best_loss = new_loss
    
    if(plot):
        print('Generating plot ' + opt_params['plot_file'])
        plt.figure(0)
        x_axis = range(0, num_iters)
        y_axis = losses
        loss_plot = plt.plot(x_axis, y_axis)
        plt.savefig(opt_params['plot_file'])
        plt.close()
    
    return best_x

def evaluate_loss(obj_function, batch_size, x, args):    
    X = args[0]
    y = args[1]

    batch_size_loss = min(X.shape[0], batch_size * 10)
    idx = np.random.choice(X.shape[0], size=batch_size, replace=False)
    X_batch = X[idx, :]
    y_batch = y[idx]

    new_args = list(args)
    new_args[0] = X_batch
    new_args[1] = y_batch
    new_args = tuple(new_args)
    return obj_function(x, *new_args)

def evaluate_gradient(gradient, batch_size, x, args):
    X = args[0]
    y = args[1]
    
    idx = np.random.choice(X.shape[0], size=batch_size, replace=False)
    X_batch = X[idx, :]
    y_batch = y[idx]
    
    new_args = list(args)
    new_args[0] = X_batch
    new_args[1] = y_batch
    new_args = tuple(new_args)
        
    return gradient(x, *new_args)
    
