import numpy as np
from scipy import spatial

def generate_argmax_data(k, d, num_samples):
    W = generate_class_vectors(k, d)

    X = np.zeros([num_samples, d])
    Y = np.zeros([num_samples, 1])

    mean = [0] * d
    cov = np.eye(d)
    for i in range(0, num_samples):
        vec = np.random.multivariate_normal(mean, cov)
        X[i, ] = vec

        true_y = get_closest_vector(W, vec, k)
        r = np.random.uniform()
        if(r < 0.8):
            Y[i, ] = true_y
        else:
            Y[i, ] = np.random.randint(0, k)

    return (W, X, Y)


def generate_softmax_data(k, d, num_samples):
    W = generate_class_vectors(k, d)

    X = np.zeros([num_samples, d])
    Y = np.zeros([num_samples, 1])

    cov = np.eye(d) * (get_lowest_vector_distance(W) / (1000 * k))
    for i in range(0, num_samples):
        true_y = np.random.randint(0, k)

        vec = np.random.multivariate_normal(W[true_y, ], cov)
        X[i, ] = vec
        Y_dist = np.exp(np.matmul(W, vec))
        Y_dist = Y_dist / np.sum(Y_dist)
        label = np.random.choice(np.arange(0, k), p=Y_dist)
        Y[i, ] = label


    return (W, X, Y)


def generate_class_vectors(k, d):
    W = np.ones([k, d])
    # Generate w vectors for classes
    mean = [0] * d
    cov = np.eye(d)
    for j in range(0, k):
        vec = np.random.multivariate_normal(mean, cov)
        vec = vec / np.linalg.norm(vec)
        W[j, ] = vec

    return W


def get_lowest_vector_distance(W):
    arr = spatial.distance.cdist(W, W)
    return np.min(arr[np.nonzero(arr)])


def get_closest_vector(W, vec, k):
    best_distance = float('inf')
    for i in range(0, k):
        dist = np.linalg.norm(W[i, ] - vec)
        if(dist < best_distance):
            best_distance = dist
            y = i
    return y

def main():
    (W, X, Y, mistakes, true_Y) = generate_softmax_data(15, 10, 5)
    return 1

if __name__ == '__main__':
    main()