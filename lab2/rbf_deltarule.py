import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

def datashuffler(train, test, target_1, test_target_1, target_2, test_target_2):


    # Shuffle data
    index_train = np.random.permutation(len(train))
    index_test = np.random.permutation(len(test))
    #np.random.shuffle(train)
    train = train[index_train]
    test=test[index_test]
    target_1 = target_1[index_train]
    test_target_1=test_target_1[index_test]
    target_2 = target_2[index_train]
    test_target_2=test_target_2[index_test]

    return (train, test, target_1, test_target_1, target_2, test_target_2)


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
    test_target_1 = sin_function(test)
    target_2 = square_function(train)
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

def phi_vector(xi, mu, sigma):
    phi_vector = np.zeros(len(mu))
    for i in range(len(mu)):
        phi_vector[i] = phi_function(xi, mu[i], sigma[i])
    return phi_vector

def error_function(target, phi_vector, weights):
    error= (target - np.matmul(phi_vector,np.transpose(weights)))
    # print(error)
    #error = (target - np.dot(weights,phi_vector.T))
    return error

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

def init_mus(nodes_number, train):
    mus = np.zeros(nodes_number)

    chunks = chunkify(train, nodes_number)
    for i,elem in enumerate(chunks):
        mean = np.mean(elem)
        mus[i] = mean
    return mus

def rnd_init_mus(nodes_number, train):
    mus = (np.random.permutation(train))[0:(nodes_number-1)]
    return mus

def main():
    eta = 0.0001
    sigma_value=0.2
    nodes= 60


    for nodes in [27, 40, 50, 63]:#[10, 20, 30, 40, 50, 63]:
        errors_sin_train = []
        errors_square_train = []
        errors_sin_test = []
        errors_square_test = []
        for sigma_value in [0.2]:# [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 1]:
            for typ in ['sin', 'square']:
                #for mutyp in ['rnd', 'mean']:
                for mutyp in ['mean']:

                    noise=1 #noise=0 without noise, noise=1 for a gaussian noise
                    train, test, target_1, target_2, test_target_1, test_target_2 = data(noise)
                    #mu= np.linspace(0,2*np.pi,nodes)
                    if mutyp == 'rnd':
                        mu = rnd_init_mus(nodes, train)
                    elif mutyp == 'mean':
                        mu = init_mus(nodes, train)
                    else:
                        print("WRONG MUTYP")
                        exit(1)
                    sigma = np.ones(len(mu)) * sigma_value
                    weights = weights_init(mu)
                    # plt.plot(target_2)
                    # plt.show()
                    error_sum = 0
                    #errors = []

                    #errors_test = []
                    epochs = 2000
                    phi_vecs =[]
                    phi_vecs_test=[]

                    for i in range(epochs):
                        train, test, target_1, test_target_1, target_2, test_target_2 = datashuffler(train, test, target_1, test_target_1, target_2, test_target_2)

                        sumerror_sin = 0
                        sumerror_square=0
                        sumerror_sin_test = 0
                        sumerror_square_test=0

                        for j in range(len(train)):
                            phi = phi_vector(train[j], mu, sigma)
                            phi_test= phi_vector(test[j], mu, sigma)
                            if i==epochs-1:
                                phi_vecs.append(phi)
                                phi_vecs_test.append(phi_test)
                            if typ == 'sin':
                                target = target_1
                                target_test= test_target_1
                                error_sin_train = error_function(target[j], phi, weights)
                                error= error_sin_train
                                error_sin_test = error_function(target_test[j], phi, weights)
                            if typ == 'square':
                                target = target_2
                                target_test = test_target_2
                                error_square_train = error_function(target[j], phi, weights)
                                error_square_test = error_function(target_test[j], phi, weights)
                                error = error_square_train

                            deltaW = eta*error*phi
                            weights = weights + deltaW
                            if typ == 'sin':
                                sumerror_sin += (1/2)*error_sin_train**2
                                sumerror_sin_test += (1 / 2) * error_sin_test ** 2
                            if typ == 'square':
                                sumerror_square += (1 / 2) * error_square_train ** 2
                                sumerror_square_test += (1 / 2) * error_square_test ** 2
                            # e =  np.sqrt((np.sum(error*error))) / len(error)

                        if typ == 'sin':
                            errors_sin_train.append(sumerror_sin / len(train))
                            errors_sin_test.append(sumerror_sin_test / len(test))
                        if typ == 'square':
                            errors_square_train.append(sumerror_square / len(train))
                            errors_square_test.append(sumerror_square_test / len(test))

                    if typ == 'sin':
                        print("type:", typ, "mus init type:", mutyp ,"sigma_value:", sigma_value, "nodes:", nodes, "error:",sumerror_sin/len(train))
                        print("type:", typ, "mus init type:", mutyp, "sigma_value:", sigma_value, "nodes:", nodes, "error:",sumerror_sin_test / len(test))
                    if typ == 'square':
                        print("type:", typ, "mus init type:", mutyp ,"sigma_value:", sigma_value, "nodes:", nodes, "error:",sumerror_square/len(train))
                        print("type:", typ, "mus init type:", mutyp, "sigma_value:", sigma_value, "nodes:", nodes, "error:",sumerror_square_test / len(test))
                    # #train prediction
                    # prediction = np.dot(phi_vecs,weights)
                    # name = typ +" train approximation, delta rule with nodes = "+str(nodes)
                    # plt.title(name)
                    # plt.scatter(train, prediction,s=2.5, label="Prediction")
                    # plt.scatter(train, target, s=2.5, label="Target")
                    # plt.legend()
                    # plt.show()
                    # #test prediction
                    # prediction = np.dot(phi_vecs_test, weights)
                    # name = typ + " test approximation, delta rule with nodes = " + str(nodes)
                    # plt.title(name)
                    # plt.scatter(test, prediction, s=2.5, label="Prediction")
                    # plt.scatter(test, target_test, s=2.5, label="Target")
                    # plt.legend()
                    # plt.show()

        iterations = np.arange(epochs)
        name= "Train error/iteration delta rule with nodes = "+str(nodes)
        plt.title(name)
        plt.plot(iterations, errors_sin_train, label="Sinus Error")
        plt.plot(iterations, errors_square_train, label="Square Error")
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.legend()
        plt.show()

        name = "Test error/iteration delta rule with nodes = " + str(nodes)
        plt.title(name)
        plt.plot(iterations, errors_sin_test, label="Sinus Error")
        plt.plot(iterations, errors_square_test, label="Square Error")
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.legend()
        plt.show()




if __name__ == "__main__":
    main()
