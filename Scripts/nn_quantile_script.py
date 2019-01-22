import sys
sys.path.append("..")
import numpy as np
import nn_classifiers as nc
from data_generation import generate_simplex_data
from matplotlib import pyplot as plt

import pickle

if __name__ == '__main__':
    k = 3
    n = 10000

    X_train, y_train = generate_simplex_data(k, n)
    X_test, y_test = generate_simplex_data(k, n)
    
    nn_clf = nc.NeuralNetQuantile(gamma=0.5, max_iter=10000, 
                 surrogate_type='AT', loss_function='hinge')
    
    nn_clf.fit(X_train, y_train, use_multiprocessing=True)
    
    fig, ax = plt.subplots()
    plt.scatter([X_test[:, 0]], [X_test[:, 1]], c=[nn_clf.predict(X_test)])
    ax.legend()
    plt.show()
    
    nn_clf.nn_.save('nn_clf.h5')
    