import numpy as np
from data_generation import *

def init(k):
    n = 100
    
    global X_train
    global y_train
    global X_test
    global y_test
    
    X_train, y_train = generate_simplex_data(k,n)
    X_test, y_test = generate_simplex_data(k,n)
    classes = np.arange(100)

    y_train = np.array(y_train).astype(int)
    y_test = np.array(y_test).astype(int)