import sys
sys.path.append("..")
from sgd_optimize import optimize_batch

import tensorflow as tf
from tensorflow import keras
from sklearn import base
import numpy as np
import copy
# from multiprocessing import Pool, Manager

def sigmoid(t):
    # sigmoid function, 1 / (1 + exp(-t))
    # stable computation
    idx = t > 0
    out = np.zeros_like(t)
    out[idx] = 1. / (1 + np.exp(-t[idx]))
    exp_t = np.exp(t[~idx])
    out[~idx] = exp_t / (1. + exp_t)
    return out

def log_loss_np(Z):
    # stable computation of the logistic loss
    idx = Z > 0
    out = np.zeros_like(Z)
    out[idx] = np.log(1 + np.exp(-Z[idx]))
    out[~idx] = (-Z[~idx] + np.log(1 + np.exp(Z[~idx])))
    return out

def hinge_loss_np(Z):
    out = np.maximum(1 - Z, 0)
    return out

def hinge_derive(t):
    idx = t > -1
    out = np.zeros_like(t)
    out[idx] = 1
    out[~idx] = 0
    return out

def hinge_loss(score):
    return tf.maximum(tf.subtract(tf.ones_like(score), score), tf.zeros_like(score))

def logistic_loss(score):
    return tf.log1p(tf.exp(tf.negative(score)))

def mode(array):
    most = max(list(map(array.count, array)))
    return list(set(filter(lambda x: array.count(x) == most, array)))

def get_surrogate_loss(gamma, thetas, n_class, surrogate_type, loss_function):
    if(loss_function == 'hinge'):
        lf = hinge_loss
    elif(loss_function == 'logistic'):
        lf = logistic_loss
        
    if(surrogate_type == 'IT'):
        loss_fd = (1 - gamma) * np.diag(np.ones(n_class - 1)) + \
                  gamma * np.diag(np.ones(n_class - 2), k=-1)
        loss_fd = np.vstack((loss_fd, np.zeros(n_class - 1)))
        loss_fd[-1, -1] = gamma 
    elif(surrogate_type == 'AT'):
        loss_fd = np.ones((n_class, n_class - 1)) * (1 - gamma)
        lower_indices = np.tril_indices(n = n_class, k = -1)
        loss_fd[lower_indices] = gamma
    
    thetas = tf.cast(thetas, tf.float32)
    thetas_tf = tf.convert_to_tensor(thetas)
    def surrogate_loss(predicted_y, desired_y):
        
        ## Compute alphas from thetas and predicted_y
        Alpha = tf.subtract(thetas_tf, predicted_y)

        ## Compute signs from desired_y
        S = tf.sign(tf.subtract(tf.cast(tf.range(n_class - 1), tf.float32), 
                                            tf.cast(desired_y, tf.float32)) + 0.5)
        
        ## Get forward difference for loss function for each sample
        loss_fd_tf = tf.cast(tf.convert_to_tensor(loss_fd), tf.float32)
        indices = tf.reshape(desired_y, [-1])
        indices = tf.minimum(indices, (n_class - 1) * tf.ones_like(indices))
        indices = tf.maximum(indices, tf.zeros_like(indices))
        loss_fd_tf = tf.gather(loss_fd_tf, tf.cast(indices, tf.int32))
        
        ## Compute loss function from loss_fd, signs, and alphas
        err = tf.matmul(tf.transpose(loss_fd_tf), lf(tf.multiply(S,Alpha)))
        obj = tf.reduce_mean(err)
        
        return obj
    
    return surrogate_loss


def weighted_absolute_loss(gamma):
    def weighted_lf(predicted_y, desired_y):
        return tf.reduce_mean((1 - gamma) * tf.maximum(tf.subtract(predicted_y, desired_y) , tf.zeros_like(desired_y))
                              + gamma * tf.maximum(tf.subtract(desired_y, predicted_y), tf.zeros_like(desired_y)))
    return weighted_lf
    
def optimize_model(thetas, X, y, theta0, n_class, gamma, max_iter, surrogate_type, loss_function, use_multiprocessing):
    num_samples, num_dimensions = X.shape
    training_loss = get_surrogate_loss(gamma, thetas, n_class, surrogate_type, loss_function)

    num_embeddings = 128
    model = keras.Sequential([
        keras.layers.Dense(num_dimensions, activation=tf.nn.relu, input_shape=(num_dimensions,)),
        keras.layers.Dense(num_embeddings, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.relu)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(), 
                  loss=training_loss,
                  metrics=['accuracy', weighted_absolute_loss(gamma)])

    model.fit(X, y, epochs= int(max_iter / num_samples), use_multiprocessing=use_multiprocessing)
    
    return model
    
def obj_margin(x0, preds, y, model, n_class, weights, L, loss_function):
    """
    Objective function for thresholds the general margin-based formulation
    """
    theta = x0
    loss_fd = weights[y.ravel().astype(int)]

    Alpha = theta - preds  # (n_class - 1, n_samples)
    S = np.sign(np.arange(n_class - 1) - y + 0.5)

    if(loss_function == 'logistic'):
        lf = log_loss_np
    elif(loss_function == 'hinge'):
        lf = hinge_loss_np
        
    err = np.dot(loss_fd.T, lf(np.multiply(-S, Alpha)))
    obj = np.mean(err)
    
    return obj

def grad_margin(x0, preds, y, model, n_class, weights, L, loss_function):
    """
    Gradient for thresholds in general margin-based formulation
    """
    theta = x0
    loss_fd = weights[y.ravel().astype(int)]

    Alpha = theta - preds # (n_class - 1, n_samples)
    S = np.sign(np.arange(n_class - 1) - y + 0.5)

    if(loss_function == 'logistic'):
        lf = sigmoid
    elif(loss_function == 'hinge'):
        lf = hinge_derive
        
    Sigma = np.dot(S, loss_fd.T)
    Sigma = np.dot(Sigma, lf(np.multiply(-S, Alpha)))

    grad_theta = -Sigma.sum(0)
    return grad_theta

def optimize_thetas(model, old_thetas, X, y, n_class, gamma, max_iter, surrogate_type, loss_function):
    curr_preds = model.predict(X)
    if(surrogate_type == 'IT'):
        loss_fd = (1 - gamma) * np.diag(np.ones(n_class - 1)) + \
                  gamma * np.diag(np.ones(n_class - 2), k=-1)
        loss_fd = np.vstack((loss_fd, np.zeros(n_class - 1)))
        loss_fd[-1, -1] = gamma 
    elif(surrogate_type == 'AT'):
        loss_fd = np.ones((n_class, n_class - 1)) * (1 - gamma)
        lower_indices = np.tril_indices(n = n_class, k = -1)
        loss_fd[lower_indices] = gamma
        
    L = np.zeros((n_class - 1, n_class - 1))
    L[np.tril_indices(n_class-1)] = 1
    
    new_thetas = optimize_batch(obj_function=obj_margin, x0=old_thetas, method='SGD', gradient=grad_margin, 
                   opt_params={'learning_rate': 1e-4, 'batch_size': 10}, bounds=None, epsilon=1e-6, max_iters=100, 
                   min_iters=50, args=(curr_preds, y, model, n_class, loss_fd, L, loss_function))
    return new_thetas

def train_quantile_nn(X, y, theta0, n_class, gamma, max_iter, surrogate_type, loss_function, use_multiprocessing):
    sess = tf.Session()
    num_iters = 2
    thetas = theta0
    old_thetas = sess.run(thetas)
    
    best_loss = np.inf
    best_model = None
    best_thetas = None
    for i in range(0, num_iters):
        model = optimize_model(thetas, X, y, theta0, n_class, gamma, max_iter, 
                               surrogate_type, loss_function, use_multiprocessing)
        thetas = optimize_thetas(model, old_thetas, X, y, n_class, gamma, max_iter, surrogate_type, loss_function)
        
        curr_loss, curr_acc, curr_weighted = model.evaluate(X, y)
        if(curr_weighted < best_loss):
            best_model = copy.copy(model)
            best_thetas = copy.copy(thetas)
            best_loss = curr_weighted
            
        old_thetas = thetas
        
    return best_model, best_thetas

def nn_threshold_predict(X, nn, theta):
    """
    Class numbers are assumed to be between 0 and k-1
    """
    tmp = theta.reshape((1, np.size(theta))) - nn.predict(X)
    pred = np.sum(tmp < 0, axis=1).astype(np.int)
    return np.array(pred)

def nn_threshold_proba(X, nn, theta):
    """
    Class numbers are assumed to be between 0 and k-1
    """
    
    eta = theta - np.asarray(nn.predict(X), dtype=np.float64)
    prob = np.pad(
        sigmoid(eta),
        pad_width=((0, 0), (1, 1)),
        mode='constant',
        constant_values=(0, 1))
    return np.diff(prob)

class NeuralNetQuantile(base.BaseEstimator):
    """
    Classifier that implements the ordinal logistic model for quantile estimation
    """
    def __init__(self, gamma=0.5, max_iter=1000, 
                 surrogate_type='AT', loss_function='logistic'):
        self.gamma = gamma
        self.max_iter = max_iter
        self.surrogate_type = surrogate_type
        self.loss_function = loss_function

    def fit(self, X, y, use_multiprocessing=False):
        _y = np.array(y).astype(np.int)
        if np.abs(_y - y).sum() > 0.1:
            raise ValueError('y must only contain integer values')
        
        self.classes_ = np.unique(y)
        self.n_class_ = int(self.classes_.max() - self.classes_.min() + 1)
        y_tmp = y - y.min()  # we need classes that start at zero

        theta0 = tf.convert_to_tensor(np.arange(self.n_class_ - 1))
        
        self.nn_, self.theta_ = train_quantile_nn(X, y, theta0, self.n_class_, self.gamma, self.max_iter, 
                                                  self.surrogate_type, self.loss_function, use_multiprocessing)
        
        return self

    def predict(self, X):
        return nn_threshold_predict(X, self.nn_, self.theta_) +\
         self.classes_.min()
    
    def predict_proba(self, X):
        return nn_threshold_proba(X, self.nn_, self.theta_)
    
class NeuralNetQuantileMulticlass(base.BaseEstimator):
    """
    Classifier that implements the ordinal logistic model for quantile estimation

    """
    def __init__(self, surrogate_type='AT', gammas=[0.5], max_iter=1000, loss_function='logistic'):
        self.surrogate_type = surrogate_type
        self.gammas = gammas
        self.max_iter = max_iter
        self.loss_function = loss_function

    def fit(self, X, y, sample_weight=None):
#         manager = Manager()
#         classifiers = manager.dict()
#         arg_list = []

        classifiers = {}
        for i, gamma in enumerate(self.gammas):
            curr_classifier = NeuralNetQuantile(gamma=gamma, max_iter=self.max_iter, 
                 surrogate_type=self.surrogate_type, loss_function=self.loss_function)
            
            curr_classifier.fit(X, y, use_multiprocessing=False)
            classifiers[gamma] = curr_classifier
            self.n_class_ = curr_classifier.n_class_
#             arg_list.append(curr_classifier)
        
#         p = Pool()    
#         def nn_one_quantile_fit(clf):
#             return clf.fit(X, y, use_multiprocessing=False)
        
#         for result in p.imap_unordered(nn_one_quantile_fit, arg_list):
#             ## nothing needed here
#             print('Done one quantile')
        
#         p.close()
        self.classifiers = classifiers
        return self

    def predict(self, X):
        n = X.shape[0]
        s = len(self.gammas)
        all_preds = np.zeros((n, s))
        for i, gamma in enumerate(self.gammas):
            all_preds[:, i] = self.classifiers[gamma].predict(X)
            
        preds = np.zeros((n, 1))
        for i in range(1, n):
            quantiles = all_preds[i, :]
            preds[i, :] = np.random.choice(mode(quantiles.tolist()))
        
        return preds
    
    def predict_score(self, X):
        n = X.shape[0]
        k = self.n_class_
        all_scores = np.zeros((n, k))
        
        for i, gamma in enumerate(self.gammas):
            all_scores = all_scores + self.classifiers[gamma].predict_proba(X)
            
        preds = np.zeros((n, 1))
        for i in range(1, n):
            curr_scores = all_scores[i, :]
            preds[i, :] = np.argmax(curr_scores)
        
        return preds