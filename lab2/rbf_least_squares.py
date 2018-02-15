import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

def data(noise):
    train = np.arange(0, 2*np.pi, 0.1)
    test = np.arange(0.05, 2*np.pi, 0.1)
    #Adding noise with variance=0.1
    if (noise==1):
        noise_train = np.random.normal(0, 0.1, len(train))
        noise_test = np.random.normal(0, 0.1, len(test))
        train = train + noise_train
        test = test + noise_test

    target_1 = sin_function(train)
    # plt.title("sin_noise")
    # plt.plot(target_1)
    # plt.show()
    test_target_1 = sin_function(test)
    # plt.title("sin_test_noise")
    # plt.plot(test_target_1)
    # plt.show()
    target_2 = square_function(train)
    test_target_2 = square_function(test)

    return train, test, target_1, target_2, test_target_1, test_target_2

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

def error_mean_square(f, target):
    sum = 0
    for i in range(len(f)):
        sum += np.absolute((f[i] - target[i]))**2
    return sum/len(f)


def chunkify(seq, num):
    #Divide the train set in "number of nodes" equal parts. 
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

def init_mus(nodes_number, train):
    mus = np.zeros(nodes_number)

    chunks = chunkify(train, nodes_number)
    for i,elem in enumerate(chunks):
        mean = np.mean(elem)
        mus[i] = mean
    return mus

def main():
    # nodes = [2, 3]
    nodes = np.arange(2,60, 10)
    sigma_value= 0.5
    errors_sin = []
    errors_square = []
    no_nums = []
    # train, test, target_1, target_2, test_target_1, test_target_2 = data()

    for nodes_number in nodes:
        try:
            # nodes_number = 5
            print(nodes_number, ": nodes number")
            noise = 1
            train, test, target_1, target_2, test_target_1, test_target_2 = data(noise)
            mu = init_mus(nodes_number, train)
            sigma = np.ones(len(mu)) * sigma_value

            #Calculation of phi_matrices:
            phi_train = phi_matrix(train, mu, sigma)
            phi_test = phi_matrix(test, mu, sigma)

            #Train: 
            #sinus weights
            weights_1 = np.linalg.lstsq(phi_train, target_1)
            #squares weights
            weights_2 = np.linalg.lstsq(phi_train, target_2)

            #Sinus prediction
            prediction1 = np.dot(phi_test,weights_1[0])
            name = "Sinus approximation", nodes_number
            plt.title(name)
            plt.plot(test, prediction1, MarkerSize=1.5, label= "Prediction")
            plt.plot(test, test_target_1, MarkerSize=15.0, label= "Target")
            plt.legend()
            plt.show()

            #Square prediction
            prediction2 = np.dot(phi_test,weights_2[0])
            name = "Square approximation", nodes_number
            plt.title(name)
            plt.plot(test, prediction2, label= "Prediction")
            plt.plot(test, test_target_2, label= "Target")
            plt.legend()
            plt.show()

            #Errors
            error_sin = error_mean_square(prediction1, test_target_1)
            error_square = error_mean_square(prediction2, test_target_2)
            print(error_sin, ': Error sinus')
            print(error_square, ': Error square')

            no_nums.append(nodes_number)
            errors_sin.append(error_sin)
            errors_square.append(error_square)

        except numpy.linalg.linalg.LinAlgError as err:
            print("Exception ", err)

    plt.title("Absolute residual error")
    plt.plot(no_nums, errors_sin, label= "Sinus Error")
    plt.plot(no_nums, errors_square, label= "Square Error")
    plt.xlabel('Number of nodes')
    plt.ylabel('Error')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

# def f_function(x, mu, sigma, weights):
#     y = np.zeros(len(x))
#     for i in range(len(x)):
#         add = 0
#         for j in range(len(mu)):
#             add += phi_function(x[i], mu[j], sigma[j])*weights[j]
#         y[i] = add
#     return y

# def weight_update_batch(phi, f):
#     #A = inv(np.matmul(np.transpose(phi_matrix), phi_matrix))
#     #w = np.matmul(np.matmul(A,np.transpose(phi_matrix)), np.transpose(f))
#     A = inv(np.matmul(np.transpose(phi), phi))
#     w = np.matmul(np.matmul(A, np.transpose(phi)), np.transpose(f))
#     return w

# def weights_init(x):
#     weights = np.random.rand(len(x))
#     return weights