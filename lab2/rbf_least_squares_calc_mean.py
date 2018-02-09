import numpy as np
import numpy.matlib
from numpy.linalg import inv
import matplotlib.pyplot as plt

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

def weight_update_batch(phi, f):
    #A = inv(np.matmul(np.transpose(phi_matrix), phi_matrix))
    #w = np.matmul(np.matmul(A,np.transpose(phi_matrix)), np.transpose(f))
    A = inv(np.matmul(np.transpose(phi), phi))
    w = np.matmul(np.matmul(A, np.transpose(phi)), np.transpose(f))
    return w

def error_mean_square(f, target):
    sum = 0
    for i in range(len(f)):
        sum += np.absolute((f[i] - target[i]))**2
    return sum/len(f)

def chunkify(seq, num):
    out = []

    if len(seq) % num == 0:
        length = len(seq) / num
        plus = 0
    else:
        length = len(seq) / num
        plus = len(seq) % num

    howmany = 0
    while howmany < len(seq):
        first = howmany
        last = first + length
        if plus > 0:
            last += 1
            plus -= 1
        out.append(seq[int(first):int(last)])
        howmany += (last - first)

    return out

    # avg = len(seq) / float(num)
    # out = []
    # last = 0.0
    # while last < len(seq):
    #     out.append(seq[int(last):int(last + avg)])
    #     last += avg
    # return out

def init_mus(nodes_number, train):
    mus = np.zeros(nodes_number)

    chunks = chunkify(train, nodes_number)
    for i,elem in enumerate(chunks):
        mean = np.mean(elem)
        mus[i] = mean
    return mus

def init_sigmas(nodes_number, train):
    sigmas = np.zeros(nodes_number)
    chunks = chunkify(train, nodes_number)
    for i, elem in enumerate(chunks):
        sigmas[i] = np.var(elem)
    return sigmas

def main():
    nodes = np.arange(2,60, 10)

    errors_train = []
    errors_test = []
    no_nums = []

    for nodes_number in nodes:
        try:
            # nodes_number = 5
            print(nodes_number, ": nodes number")
            train, test, target_1, target_2, test_target_1, test_target_2 = data()
            mu = init_mus(nodes_number, train)
            sigma = np.ones(len(mu)) * 0.5
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

            plt.plot(train, target_1)
            plt.plot(train, output_train)
            plt.plot(test, output_test)
            plt.show()
            plt.clf()

            no_nums.append(nodes_number)
            errors_test.append(error_test)
            errors_train.append(error_train)




        except numpy.linalg.linalg.LinAlgError as err:
            print("Exception ", err)


    plt.plot(no_nums, errors_test)
    plt.plot(no_nums, errors_train)
    plt.show()


                                                                                #Comment: Weird, test is better than train. 

if __name__ == "__main__":
    main()