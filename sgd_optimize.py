import numpy as np
import matplotlib.pyplot as plt

def optimize_batch(obj_function=None, x0=None, method='SGD', gradient=None, 
                   opt_params={'learning_rate': 1, 'momentum_gamma': 0.9, 'beta1': 0.9, 'beta2': 0.99},
                   bounds=None, epsilon=1e-6, max_iters=10000, min_iters=500, batch_size=100, args=None, plot=True):
    x = x0
    best_x = x0
    best_loss = 1e8
    
    num_iters = 0
    loss_difference = 1e8
    new_loss = 1e8
    
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
            eta = 1 / (1 + learning_rate * num_iters)
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
            
        new_loss = evaluate_loss(obj_function, x, args)
        
        if(plot):
            losses.append(new_loss)
        
        loss_difference = np.abs(new_loss - old_loss) / old_loss
        
        num_iters += 1
        
        if(new_loss < best_loss):
            best_x = x
            best_loss = new_loss
    
    if(plot):
        plt.figure(0)
        x_axis = range(300, num_iters)
        y_axis = np.log(losses[300:])
        loss_plot = plt.plot(x_axis, y_axis)
        plt.savefig(opt_params['plot_file'])
        plt.close()
    
    return best_x

def evaluate_loss(obj_function, x, args):    
    return obj_function(x, *args)

def evaluate_gradient(gradient, batch_size, x, args):
    X = args[0]
    y = args[1]
    
    idx = np.random.randint(X.shape[0], size=batch_size)
    X_batch = X[idx, :]
    y_batch = y[idx]
    
    new_args = list(args)
    new_args[0] = X_batch
    new_args[1] = y_batch
    new_args = tuple(new_args)
        
    return gradient(x, *new_args)
    
