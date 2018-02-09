import numpy as np
import numpy.matlib
from numpy.linalg import inv

def data():
    train = np.arange(0, 2*np.pi, 0.1)
    test = np.arange(0.05, 2*np.pi, 0.1)
    target_1 = sin_function(train)
    target_2 = square_function(train)
    return train, test, target_1, target_2

def weights_init(x):
    weights = np.random.rand(len(x))
    return weights

def sin_function(x):
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = np.sin(2*x[i])
    return y

def square_function(x):
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = np.sign(np.sin(2*x[i]))
    return y

def phi_function(x, mu, sigma):
    phi = np.exp(-(x-mu)**2/(2*sigma**2))
    return phi

def phi_matrix(x, mu, sigma):
    phi = np.zeros((len(x), len(mu)))
    for i in range(len(x)):
        for j in range(len(mu)):
            phi[i][j] = phi_function(x[i], mu[j], sigma[j])
    return phi

def f_function(x, mu, sigma, weights):
    y = np.zeros(len(x))
    for i in range(len(x)):
        add = 0
        for j in range(len(mu)):
            add += phi_function(x[i], mu[j], sigma[j])*weights[j]
        y[i] = add
    return y

def weight_update_batch(phi_matrix, f): 
    A = inv(np.matmul(np.transpose(phi_matrix), phi_matrix))
    w = np.matmul(np.matmul(A,np.transpose(phi_matrix)), np.transpose(f))
    return w

def error_mean_square(f, target):
    sum = 0
    for i in range(len(f)):
        sum += (f[i] - target[i])**2
    return sum

def main():
    train, test, target_1, target_2 = data()
    # mu = np.array([0.3, 0.01, 0.5])
    # sigma = np.array([0.5, 10, 1])
    mu = train
    sigma = np.ones(len(train))*0.5
    weights = weights_init(mu)

    phi = phi_matrix(train, mu, sigma)
    for i in range(100):
        f = f_function(train, mu, sigma, weights)
        weights = weight_update_batch(phi, f)
        error = error_mean_square(f, target_1)
        print(error)

if __name__ == "__main__":
    main()