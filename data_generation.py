import numpy as np
import scipy

## Create one linear classifier, and k-1 "barriers" such that,
## if point x lies between barrier i and i-1, the probability it belongs to class i
## is proportional to how close it is to that barrier
def generate_quantile_data(d, k, n, portions):
    ## Vector that classification will be based on
    w = np.array([1]*d)
    scaling = portions[-1]

    ## Generate points
    projection_points = np.matrix(np.random.uniform(0, scaling, n)).T
    points_on_line = projection_points * w

    #Ensure noise in range as to not mess with classification too much
    noise = np.reshape(np.matrix(np.random.normal(0, 1, n*d)), (n, d))
    noise = noise - np.dot(noise, w).T * w
    X = points_on_line + noise

    ## Get class labels
    y = get_class_labels(projection_points, portions, k)
    
    return X,y

def get_class_labels(projection_points, portions, k):
    y = np.matrix(np.zeros(projection_points.size)).T
    for i in range(0, projection_points.size):
        curr_point = projection_points[i]
        likely_class = np.argmax((curr_point - portions) < 0)
        if(likely_class == k - 1):
            y[i] = k-1
        else:
            if(likely_class > 0):
                prob_up = 1 - (portions[likely_class] - curr_point) / (portions[likely_class] - portions[likely_class - 1])
            else:
                prob_up = 1 - (portions[likely_class] - curr_point) / (portions[likely_class] - 0)
            y[i] = likely_class + 1 * (np.random.uniform(0,1) < prob_up)
    return y

def generate_linear_complex_data(d, k, n, portions):
    ## Vector that classification will be based on
    w = np.array([1/d]*d)
    scaling = portions[-1]

    ## Generate points
    projection_points = np.matrix(np.random.uniform(0, scaling, n)).T
    points_on_line = projection_points * w

    #Ensure noise in range as to not mess with classification too much
    noise = np.reshape(np.matrix(np.random.normal(0, scaling / (16*d), n*d)), (n, d))
    noise = noise - np.dot(noise, w).T * w
    X = points_on_line + noise

    ## Get class labels
    y = get_many_class_labels(projection_points, portions, k)
    
    return X,y

def get_many_class_labels(projection_points, portions, k):
    y = np.matrix(np.zeros(projection_points.size)).T
    for i in range(0, projection_points.size):
        curr_point = projection_points[i]
        probs = 1 / np.power(np.abs(portions - curr_point), 2)
        probs = np.asarray(probs / np.sum(probs))
        y[i] = np.random.choice(k, p=probs[0])
    return y

## Create one linear classifier, and k-1 "barriers" such that,
## if point x lies between barrier i and i-1, the probability it belongs to class i
## is proportional to how close it is to that barrier
def generate_quantile_testdata(d, k, n, portions):
    ## Vector that classification will be based on
    w = np.array([1/d]*d)
    scaling = portions[-1]

    ## Generate points
    projection_points = np.matrix(np.random.uniform(0, scaling, n)).T
    print(projection_points)
    points_on_line = projection_points * w
    print(points_on_line)

    #Ensure noise in range as to not mess with classification too much
    noise = np.reshape(np.matrix(np.random.normal(0, scaling / (16*d), n*d)), (n, d))
    noise = noise - np.dot(noise, w).T * w
    X = points_on_line + noise
    print(X.dot(w) *d)
    
    return X

def generate_quantile_complex_testdata(d, k, n, alpha, portions):
    ## Vector that classification will be based on
    w = np.array([1/d]*d)
    scaling = portions[-1]

    ## Generate points
    projection_points = np.matrix(np.random.uniform(0, scaling, n)).T
    points_on_line = projection_points * w

    #Ensure noise in range as to not mess with classification too much
    noise = np.reshape(np.matrix(np.random.normal(0, scaling / (16*d), n*d)), (n, d))
    noise = noise - np.dot(noise, w).T * w
    X = points_on_line + noise
    
    y = get_many_class_quantiles(projection_points, portions, k, alpha)
    
    return X, y

def get_many_class_quantiles(projection_points, portions, k, alpha):
    y = np.matrix(np.zeros(projection_points.size)).T
    for i in range(0, projection_points.size):
        curr_point = projection_points[i]
        probs = 1 / np.power(np.abs(portions - curr_point), 2)
        probs = np.asarray(probs / np.sum(probs))
        y[i] = np.argmax(np.cumsum(probs) > alpha)
    return y

## By using the uniform distribution, we can generate uniform spacings on the unit interval
## (see di Finetti's theorem)
def generate_simplex_data(k, n):
    X = np.random.uniform(size=[n, k-1]) * 10
    X = np.sort(X)
    
    classes = np.random.uniform(size=[n,1]) * 10
    y = np.zeros_like(classes)
    for i in range(0, n):
        curr_vec = X[i,:]
        pointer = classes[i,0]
        
        if(pointer < curr_vec[0]):
            y[i, 0] = 0
        elif(pointer > curr_vec[-1]):
            y[i,0] = k - 1
        else:
            y[i,0] = np.argmax(curr_vec > classes[i,0])
        
    return X, y

## Methods for simplex dataset
def compute_alpha_quantile(X, alpha):
    k = 3
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

def weighted_absolute_loss(u, y, alpha):
    y = np.array(y.T[0]) ## y is given as a column matrix, but predictions are not
    zs = np.zeros_like(y)
    return np.mean((1 - alpha) * np.maximum((u - y), zs) + alpha * np.maximum((y - u), zs))
