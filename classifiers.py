import numpy as np
from scipy import optimize
import scipy
from sklearn import base, metrics
from sklearn.utils.validation import check_X_y
from sgd_optimize import *
from multiprocessing import Pool, Manager


"""
This implements the margin-based ordinal regression methods described
in http://arxiv.org/abs/1408.2327
"""

def sigmoid(t):
    # sigmoid function, 1 / (1 + exp(-t))
    # stable computation
    idx = t > 0
    out = np.zeros_like(t)
    out[idx] = 1. / (1 + np.exp(-t[idx]))
    exp_t = np.exp(t[~idx])
    out[~idx] = exp_t / (1. + exp_t)
    return out


def log_loss(Z):
    # stable computation of the logistic loss
    idx = Z > 0
    out = np.zeros_like(Z)
    out[idx] = np.log(1 + np.exp(-Z[idx]))
    out[~idx] = (-Z[~idx] + np.log(1 + np.exp(Z[~idx])))
    return out

def hinge_loss(Z):
    out = np.maximum(1 - Z, 0)
    return out

def obj_margin(x0, X, y, alpha, n_class, weights, L, kernel_type, loss_function, sample_weight):
    """
    Objective function for the general margin-based formulation
    """

    w = x0[:X.shape[1]]
    c = x0[X.shape[1]:]
    theta = L.dot(c)
    loss_fd = weights[y]

    Xw = X.dot(w)
    Alpha = theta[:, None] - Xw  # (n_class - 1, n_samples)
    S = np.sign(np.arange(n_class - 1)[:, None] - y + 0.5)

    if(loss_function == 'logistic'):
        err = loss_fd.T * log_loss(S * Alpha)
    elif(loss_function == 'hinge'):
        err = loss_fd.T * hinge_loss(S * Alpha)
        
    if sample_weight is not None:
        err *= sample_weight
    obj = np.sum(err)
    
    ## Regularization based on kernel 
    if(kernel_type == 'linear'):
        obj += alpha * (0.5 / X.shape[1]) * (np.dot(w, w))
    else:
        obj += alpha * (0.5 / X.shape[1]) * (np.dot(w, w)) #np.dot(np.dot(w.T, X).T, w)
    
    return obj

def obj_direct(x0, X, y, alpha, gamma, n_class, weights, L, kernel_type, sample_weight):
    """
    Objective function for the direct minimization of weighted error
    """

    w = x0[:X.shape[1]]
    c = x0[X.shape[1]:]
    theta = L.dot(c)
    loss_fd = weights[y]

    tmp = theta[:, None] - np.asarray(X.dot(w))
    preds = np.sum(tmp < 0, axis=0).astype(np.int)

    zs = np.zeros_like(y)
    err = (1 - gamma) * np.maximum(preds - y, zs) + gamma * np.maximum(y - preds, zs)
    if sample_weight is not None:
        err *= sample_weight
    obj = np.sum(err)
    
    ## TODO: Add regularization
    return obj

def grad_direct(x0, X, y, alpha, gamma, n_class, weights, L, kernel_type, sample_weight):
    """
    Gradient for the direct minimization of weighted error
    """

    w = x0[:X.shape[1]]
    c = x0[X.shape[1]:]
    theta = L.dot(c)
    loss_fd = weights[y]

    tmp = theta[:, None] - np.asarray(X.dot(w))
    preds = np.sum(tmp < 0, axis=0).astype(np.int)
    
    S = np.sign(np.arange(n_class - 1)[:, None] - y + 0.5)

    zs = np.zeros_like(y)
    Sigma = S * ((1 - gamma) * (1 * np.maximum(preds - y, zs) > 0) + gamma * (1 * np.maximum(y - preds, zs) > 0))
    if sample_weight is not None:
        Sigma *= sample_weight

    grad_w = X.T.dot(Sigma.sum(0))

    grad_theta = -Sigma.sum(1)
    grad_c = L.T.dot(grad_theta)
    return np.concatenate((grad_w, grad_c), axis=0)



def hinge_derive(t):
    idx = t > -1
    out = np.zeros_like(t)
    out[idx] = 1
    out[~idx] = 0
    return out


def grad_margin(x0, X, y, alpha, n_class, weights, L, kernel_type, loss_function, sample_weight):
    """
    Gradient for the general margin-based formulation
    """

    w = x0[:X.shape[1]]
    c = x0[X.shape[1]:]
    theta = L.dot(c)
    loss_fd = weights[y]

    Xw = X.dot(w)
    Alpha = theta[:, None] - Xw  # (n_class - 1, n_samples)
    S = np.sign(np.arange(n_class - 1)[:, None] - y + 0.5)

    if(loss_function == 'logistic'):
        Sigma = S * loss_fd.T * sigmoid(-S * Alpha)
    elif(loss_function == 'hinge'):
        Sigma = S * loss_fd.T * hinge_derive(-S * Alpha)
    if sample_weight is not None:
        Sigma *= sample_weight

    if(kernel_type == 'linear'):
        grad_w = X.T.dot(Sigma.sum(0)) + (1.0 / X.shape[1]) * alpha * w
    else:
        ## Adjusted for batch SGD
        grad_w = X.T.dot(Sigma.sum(0)) + (1.0 / X.shape[1]) * alpha * w
        #grad_w = X.T.dot(Sigma.sum(0)) + 0.5 * alpha * (np.matmul(X.T, w) + np.matmul(X, w))

    grad_theta = -Sigma.sum(1)
    grad_c = L.T.dot(grad_theta)
    return np.concatenate((grad_w, grad_c), axis=0)


def threshold_fit_quantile(X, y, alpha, gamma, n_class, kernel_type, loss_function,
                           opt_type, opt_params, mode='QE', max_iter=1000, 
                           verbose=False, tol=1e-12, sample_weight=None):
    """
    Solve the general threshold-based ordinal regression model
    using the logistic loss as surrogate of the 0-1 loss

    Parameters
    ----------
    mode : string, one of {'AE', '0-1', 'Direct'}

    """

    X, y = check_X_y(X, y, accept_sparse='csr')
    unique_y = np.sort(np.unique(y))
    if not np.all(unique_y == np.arange(unique_y.size)):
        raise ValueError(
            'Values in y must be %s, instead got %s'
            % (np.arange(unique_y.size), unique_y))

    n_samples, n_features = X.shape

    # convert from c to theta
    L = np.zeros((n_class - 1, n_class - 1))
    L[np.tril_indices(n_class-1)] = 1.

    if mode == '0-1' or mode == 'Direct':
        loss_fd = (1 - gamma) * np.diag(np.ones(n_class - 1)) + \
                  gamma * np.diag(np.ones(n_class - 2), k=-1)
        loss_fd = np.vstack((loss_fd, np.zeros(n_class - 1)))
        loss_fd[-1, -1] = gamma  # border case
    elif mode == 'AE':
        # loss forward difference
        loss_fd = np.ones((n_class, n_class - 1)) * (1 - gamma)
        lower_indices = np.tril_indices(n = n_class, k = -1)
        loss_fd[lower_indices] = gamma
    else:
        raise NotImplementedError

    x0 = np.zeros(n_features + n_class - 1)
    x0[X.shape[1]:] = np.arange(n_class - 1)
    options = {'maxiter' : max_iter, 'disp': verbose}
    if n_class > 2:
        bounds = [(None, None)] * (n_features + 1) + \
                 [(0, None)] * (n_class - 2)
    else:
        bounds = None

    if (mode != 'Direct'):
        if(opt_type == 'SciPy'):
            sol = optimize.minimize(obj_margin, x0, method='L-BFGS-B',
            bounds=bounds, options=options, 
            args=(X, y, alpha, n_class, loss_fd, L, kernel_type, loss_function, sample_weight),
            tol=tol)
            sol = sol.x
        else: 
            sol = optimize_batch(obj_function=obj_margin, x0=x0, method=opt_type, 
                         bounds=bounds, gradient=grad_margin, opt_params=opt_params,
                         args=(X, y, alpha, n_class, loss_fd, L, kernel_type, loss_function, sample_weight))
    else:
        sol = optimize.minimize(obj_direct, x0, method='L-BFGS-B',
            jac=grad_direct, bounds=bounds, options=options,
            args=(X, y, alpha, gamma, n_class, loss_fd, L, kernel_type, loss_function, sample_weight),
            tol=tol)
    
    if verbose and not sol.success:
        print(sol.message)

    w, c = sol[:X.shape[1]], sol[X.shape[1]:]
    theta = L.dot(c)
    return w, theta


def threshold_predict(X, w, theta):
    """
    Class numbers are assumed to be between 0 and k-1
    """
    tmp = theta[:, None] - np.asarray(X.dot(w))
    pred = np.sum(tmp < 0, axis=0).astype(np.int)
    return pred


def threshold_proba(X, w, theta):
    """
    Class numbers are assumed to be between 0 and k-1. Assumes
    the `sigmoid` link function is used.
    """
    eta = theta[:, None] - np.asarray(X.dot(w), dtype=np.float64)
    prob = np.pad(
        sigmoid(eta).T,
        pad_width=((0, 0), (1, 1)),
        mode='constant',
        constant_values=(0, 1))
    return np.diff(prob)

def transform_kernel(X, train_set, kernel_type, kernel_param):
    if(kernel_type == 'poly'):
        K = metrics.pairwise.polynomial_kernel(X, train_set, degree=kernel_param)
        K = 1000 * K / np.linalg.norm(K)
    elif(kernel_type == 'rbf'):
        K = metrics.pairwise.rbf_kernel(X, train_set, gamma=kernel_param)
    else:
        K = X
    return K

def quantile_fit_star(args):
    return quantile_fit(*args)

def quantile_fit(clf, X, y, quantile):
    return (quantile, clf.fit(X, y))

def mode(array):
    most = max(list(map(array.count, array)))
    return list(set(filter(lambda x: array.count(x) == most, array)))

class QuantileMulticlass(base.BaseEstimator):
    """
    Classifier that implements the ordinal logistic model for quantile estimation

    """
    def __init__(self, surrogate='AT', gammas=[0.5], alphas=[10], verbose=0, max_iter=10000, 
                 kernel_type='linear', kernel_params=[1], loss_function='logistic',
                 opt_type='SGD', opt_params={}):
        self.surrogate = surrogate
        self.gammas = gammas
        self.alphas = alphas
        self.verbose = verbose
        self.max_iter = max_iter
        self.kernel_type = kernel_type
        self.kernel_params = kernel_params
        self.loss_function = loss_function
        self.opt_type = opt_type
        self.opt_params = opt_params

    def fit(self, X, y, sample_weight=None):
        _y = np.array(y).astype(np.int)
        if np.abs(_y - y).sum() > 0.1:
            raise ValueError('y must only contain integer values')
        self.classes_ = np.unique(y)
        self.n_class_ = self.classes_.max() - self.classes_.min() + 1
        
        manager = Manager()
        classifiers = manager.dict()
        
        arg_list = []
        for i, gamma in enumerate(self.gammas):
            opt_params = self.opt_params.copy()
            opt_params['plot_file'] = 'mnist_quantiles/' + self.opt_type + self.kernel_type + str(self.kernel_param) + self.surrogate + self.loss_function + '_i' + str(i + 1) + 's' + str(len(self.gammas) + 1) + '.png'
            print(opt_params)
            if(self.surrogate == 'AT'):
                curr_classifier = QuantileAT(gamma=gamma, alpha=self.alphas[i], kernel_type=self.kernel_type, 
                                         opt_type=self.opt_type, opt_params=opt_params,
                                         kernel_param=self.kernel_params[i], loss_function=self.loss_function)
            elif(self.surrogate == 'IT'):
                curr_classifier = QuantileIT(gamma=gamma, alpha=self.alphas[i], kernel_type=self.kernel_type, 
                                         opt_type=self.opt_type, opt_params=opt_params,
                                         kernel_param=self.kernel_params[i], loss_function=self.loss_function)
            curr_args = (curr_classifier, X, y, gamma)
            arg_list.append(curr_args)
        p = Pool()    
        for result in p.imap_unordered(quantile_fit_star, arg_list):
            quantile = result[0]
            clf = result[1]
            classifiers[quantile] = clf
        
        p.close()
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

class QuantileIT(base.BaseEstimator):
    """
    Classifier that implements the ordinal logistic model for quantile estimation

    """
    def __init__(self, gamma=0.5, alpha=1., verbose=0, max_iter=1000, 
                 kernel_type='linear', kernel_param=1, loss_function='logistic',
                 opt_type = 'SGD', opt_params={'learning_rate': 1e-8}):
        self.gamma = gamma
        self.alpha = alpha
        self.verbose = verbose
        self.max_iter = max_iter
        self.kernel_type = kernel_type
        self.kernel_param = kernel_param
        self.loss_function = loss_function
        self.opt_type = opt_type
        self.opt_params = opt_params

    def fit(self, X, y, sample_weight=None):
        _y = np.array(y).astype(np.int)
        if np.abs(_y - y).sum() > 0.1:
            raise ValueError('y must only contain integer values')
        self.classes_ = np.unique(y)
        self.n_class_ = self.classes_.max() - self.classes_.min() + 1
        
        # Handle Kernel
        self.train_set = X
        K = transform_kernel(X, X, self.kernel_type, self.kernel_param)
        
        y_tmp = y - y.min()  # we need classes that start at zero
        self.coef_, self.theta_ = threshold_fit_quantile(
            K, y_tmp, self.alpha, self.gamma, self.n_class_, self.kernel_type,
            self.loss_function, self.opt_type, self.opt_params,
            mode='0-1', verbose=self.verbose, max_iter=self.max_iter,
            sample_weight=sample_weight)
        return self

    def predict(self, X):
        K = transform_kernel(X, self.train_set, self.kernel_type, self.kernel_param)
        
        return threshold_predict(K, self.coef_, self.theta_) +\
         self.classes_.min()

    def predict_proba(self, X):
        K = transform_kernel(X, self.train_set, self.kernel_type, self.kernel_param)
        
        return threshold_proba(K, self.coef_, self.theta_)

    def score(self, X, y, sample_weight=None):
        K = transform_kernel(X, self.train_set, self.kernel_type, self.kernel_param)
        
        pred = self.predict(K)
        return metrics.accuracy_score(
            pred,
            y,
            sample_weight=sample_weight)
    
class QuantileAT(base.BaseEstimator):
    """
    Classifier that implements the ordinal logistic model for quantile estimation

    """
    def __init__(self, gamma=0.5, alpha=1., verbose=0, max_iter=1000, 
                 kernel_type='linear', kernel_param=1, loss_function='logistic',
                 opt_type = 'SGD', opt_params={'learning_rate': 1e-8}):
        self.gamma = gamma
        self.alpha = alpha
        self.verbose = verbose
        self.max_iter = max_iter
        self.kernel_type = kernel_type
        self.kernel_param = kernel_param
        self.loss_function = loss_function
        self.opt_type = opt_type
        self.opt_params = opt_params

    def fit(self, X, y, sample_weight=None):
        _y = np.array(y).astype(np.int)
        if np.abs(_y - y).sum() > 0.1:
            raise ValueError('y must only contain integer values')
        self.classes_ = np.unique(y)
        self.n_class_ = self.classes_.max() - self.classes_.min() + 1
        
        # Handle Kernel
        self.train_set = X
        K = transform_kernel(X, X, self.kernel_type, self.kernel_param)
        
        y_tmp = y - y.min()  # we need classes that start at zero
        self.coef_, self.theta_ = threshold_fit_quantile(
            K, y_tmp, self.alpha, self.gamma, self.n_class_, self.kernel_type,
            self.loss_function, self.opt_type, self.opt_params, mode='AE', 
            verbose=self.verbose, max_iter=self.max_iter, sample_weight=sample_weight)
        return self

    def predict(self, X):
        K = transform_kernel(X, self.train_set, self.kernel_type, self.kernel_param)
        return threshold_predict(K, self.coef_, self.theta_) +\
         self.classes_.min()

    def predict_proba(self, X):
        K = transform_kernel(X, self.train_set, self.kernel_type, self.kernel_param)
        return threshold_proba(K, self.coef_, self.theta_)

    def score(self, X, y, sample_weight=None):
        K = transform_kernel(X, self.train_set, self.kernel_type, self.kernel_param)
        pred = self.predict(K)
        return metrics.accuracy_score(
            pred,
            y,
            sample_weight=sample_weight)
    
class DirectQuantile(base.BaseEstimator):
    """
    Classifier that implements the ordinal logistic model for quantile estimation

    """
    def __init__(self, gamma=0.5, alpha=1., verbose=0, max_iter=1000):
        self.gamma = gamma
        self.alpha = alpha
        self.verbose = verbose
        self.max_iter = max_iter

    def fit(self, X, y, sample_weight=None):
        _y = np.array(y).astype(np.int)
        if np.abs(_y - y).sum() > 0.1:
            raise ValueError('y must only contain integer values')
        self.classes_ = np.unique(y)
        self.n_class_ = self.classes_.max() - self.classes_.min() + 1
        y_tmp = y - y.min()  # we need classes that start at zero
        self.coef_, self.theta_ = threshold_fit_quantile(
            X, y_tmp, self.alpha, self.gamma, self.n_class_, self.kernel_type,
            mode='Direct', verbose=self.verbose, max_iter=self.max_iter,
            sample_weight=sample_weight)
        return self

    def predict(self, X):
        return threshold_predict(X, self.coef_, self.theta_) +\
         self.classes_.min()

    def predict_proba(self, X):
        return threshold_proba(X, self.coef_, self.theta_)

    def score(self, X, y, sample_weight=None):
        pred = self.predict(X)
        return metrics.accuracy_score(
            pred,
            y,
            sample_weight=sample_weight)
