import numpy as np
import numpy.matlib
from numpy.linalg import inv

def data():
    train = np.arange(0, 2*np.pi, 0.1)
    test = np.arange(0.05, 2*np.pi, 0.1)
    target_1 = sin_function(train)
    target_2 = square_function(train)
    test_target_1 = sin_function(test)
    test_target_2 = square_function(test)
    return train, test, target_1, target_2, test_target_1, test_target_2

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
        sum += np.absolute((f[i] - target[i]))**2
    return sum/len(f)

def chunkify(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out

def init_mus(nodes_number, train):
    mus = np.zeros(nodes_number)

    chunks = chunkify(train, nodes_number)
    for i,elem in enumerate(chunks):
        mus[i] = np.mean(elem)
    return mus

def init_sigmas(nodes_number, train):
    sigmas = np.zeros(nodes_number)
    chunks = chunkify(train, nodes_number)
    for i, elem in enumerate(chunks):
        sigmas[i] = np.var(elem)
    return sigmas

def main():
    nodes_number = 9
    train, test, target_1, target_2, test_target_1, test_target_2 = data()
    mu = init_mus(nodes_number, train)
    sigma = init_sigmas(nodes_number, train)
    # weights = weights_init(mu)

#           Training
    phi = phi_matrix(train, mu, sigma)
    weights = weight_update_batch(phi, target_1)
    output_train = f_function(train, mu, sigma, weights)
    error_train = error_mean_square(output_train, target_1)
    print(error_train, ': Error train')

#           Testing
    output_test = f_function(test, mu, sigma, weights)
    error_test = error_mean_square(output_test, test_target_1)
    print(error_test, ': Error test')

                                                                                #Comment: Weird, test is better than train. 

if __name__ == "__main__":
    main()