import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

def datashuffler(train, test, target_1, test_target_1):

    # Shuffle data
    index_train = np.random.permutation(len(train))
    index_test = np.random.permutation(len(test))
    #np.random.shuffle(train)
    train = train[index_train]
    test=test[index_test]
    target_1 = target_1[index_train]
    test_target_1=test_target_1[index_test]

    return (train, test, target_1, test_target_1)


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

    return train, test, target_1, test_target_1

def sin_function(x):
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = np.sin(2*x[i])
    return y

def phi_function(x, mu, sigma):
    phi = np.exp(-(x-mu)**2/(2*sigma**2))
    return phi

def phi_vector(xi, mu, sigma):
    phi_vector = np.zeros(len(mu))
    for i in range(len(mu)):
        phi_vector[i] = phi_function(xi, mu[i], sigma)
    return phi_vector

def error_function(target, phi_vector, weights):
    error= (target - np.matmul(phi_vector,np.transpose(weights)))
    # print(error)
    #error = (target - np.dot(weights,phi_vector.T))
    return error

def find_bmu(t, net):
    """
        Find the best matching unit for a given vector, t, in the SOM
        Returns: a (bmu, bmu_idx) tuple where bmu is the high-dimensional BMU
                 and bmu_idx is the index of this vector in the SOM
    """
    bmu_idx = []
    # set the initial minimum distance to a huge number
    min_dist = np.iinfo(np.int).max
    # calculate the high-dimensional distance between each neuron and the input
    for x in range(net.shape[0]):
        #for y in range(net.shape[1]):
        w = net[x]
        # don't bother with actual Euclidean distance, to avoid expensive sqrt operation
        sq_dist = np.sum((w - t) ** 2)
        if sq_dist < min_dist:
            min_dist = sq_dist
            #bmu_idx = np.array([x, y])
            bmu_idx = x
    # get vector corresponding to bmu_idx
    #bmu = net[bmu_idx[0], bmu_idx[1], :].reshape(m, 1)
    bmu = net[bmu_idx]
    # return the (bmu, bmu_idx) tuple
    return (bmu, bmu_idx)

def decay_radius(initial_radius, i, time_constant):
    return initial_radius * np.exp(-i / time_constant)

def decay_learning_rate(initial_learning_rate, i, epochs):
    return initial_learning_rate * np.exp(-i / epochs)

def calculate_influence(distance, radius):
    return np.exp(-distance / (2* (radius**2)))

def main():
    eta = 0.0001
    sigma_value=0.2
    nodes= 20
 
    epochs = 20
    init_learning_rate = 0.2
    init_radius = 50
    time_constant = epochs / np.log(init_radius)
    net = np.random.random((20, 1))

    noise = 0 #noise=0 without noise, noise=1 for a gaussian noise
    train, test, target_1, test_target_1 = data(noise)  # calc only sin(2x)

    #Competitative learning: 
    for i in range(epochs):
        row_p = train
        bmu, bmu_idx = find_bmu(row_p, net)
        r = decay_radius(init_radius, i, time_constant)
        l = decay_learning_rate(init_learning_rate, i, epochs)
        for x in range(net.shape[0]): # number of nodes
            w = net[x]
            w_dist = np.sum((x - bmu_idx) ** 2)                                        
            if w_dist <= r**2:
                influence = calculate_influence(w_dist, r)
                new_w = w + (l * influence * (row_p - w))
                net[x] = new_w[0]
    
    mu = net.flatten()  #initialization weights for the delta rule.
    phi_vecs = []

    #Delta rule:
    for i in range(epochs):
        for j in range(len(train)):
            phi = phi_vector(train[j], mu, sigma_value)
            error = error_function(target[j], phi, weights)
        phi_vecs.append(phi)
        

    # prediction = np.dot(phi_vecs,weights)
    # name = type +" approximation, delta rule"
    # plt.title(name)
    # plt.scatter(train, prediction,s=2.5, label="Prediction")
    # plt.scatter(train, target, s=2.5, label="Target")
    # plt.legend()
    # plt.show()
    #
    # iterations = np.arange(epochs)
    # name= "Error/iteration delta rule"
    # plt.title(name)
    # plt.plot(iterations, errors,'blue')
    # plt.xlabel('Epochs')
    # plt.ylabel('Error')
    # plt.show()


if __name__ == "__main__":
    main()
