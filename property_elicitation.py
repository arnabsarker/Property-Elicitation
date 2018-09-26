import data_generation
from scipy import optimize, stats
import numpy as np

## Returns the ith linear classifier
def learn_linear_classifier(i, s, X, Y):
    d = X.shape[1]
    w_hat = optimize.minimize(opt_function, [1/d]*d, args=(i, s, X, Y), method='Nelder-Mead').x
    if(np.linalg.norm(w_hat) < 0.00001):
        w_hat = [0]*d
    else:
        w_hat = w_hat / np.linalg.norm(w_hat)
    return w_hat

def opt_function(w_hat, *args):
    i = args[0]
    s = args[1]
    X = args[2]
    Y = args[3]
    n = X.shape[0]
    total = 0
    for j in range(0, n):
        total += (1 - i / s) * max(np.matmul(np.transpose(w_hat), X[j,]) - Y[j,], 0) + \
                (i / s) * max(Y[j,] - np.matmul(np.transpose(w_hat), X[j,]) , 0)
    return total / n + 0.01 * np.linalg.norm(w_hat, 1)


def learn_all_classifiers(s, X, Y):
    d = X.shape[0]
    W_hat = np.zeros([d, s])

    for i in range(0, s):
        W_hat[i, ] = learn_linear_classifier(i, s, X, Y)

    return W_hat

def classify_sample(W_hat, x):
    quantiles = np.matmul(np.transpose(W_hat), x)
    return stats.mode(quantiles)[0][0]


def main():
    (W, X, Y) = data_generation.generate_argmax_data(12, 10, 100)

    print(learn_linear_classifier(10, 15, X, Y))
    return 1

if __name__ == '__main__':
    main()