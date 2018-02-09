import numpy as np
import numpy.matlib

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

def phi_vector(xi, mu, sigma):
    phi_vector = np.zeros(len(mu))
    for i in range(len(mu)):
        phi_vector[i] = phi_function(xi, mu[i], sigma[i])
    return phi_vector

def error_function(target, phi_vector, weights):
    error= (target - np.dot(phi_vector.T,weights))
    return error

def main():
    eta = 0.0001
    train, test, target_1, target_2 = data()
    mu = np.array([0.3, 0.01, 0.5])
    sigma = np.array([0.5, 10, 1])
    weights = weights_init(mu)

    for i in range(len(target_1)):
        phi = phi_vector(train, mu, sigma)
        error1 = error_function(target_1, phi, weights)
        deltaW = eta * error * phi
        weights = weights + deltaW

    for i in range(len(target_2)):
        phi = phi_vector(train, mu, sigma)
        error2 = error_function(target_2, phi, weights)
        deltaW = eta * error * phi
        weights = weights + deltaW


if __name__ == "__main__":
    main()
