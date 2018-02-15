import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

def read_data_from_file(name):
    with open(name, "r") as file:
        # Read the whole file at once
        data = file.read()
        data = data.split('\n')
        train = []
        train_target = []
        for line in data:
            dt = line.split('\t')
            train.append([float(i) for i in dt[0].split(' ')])
            train_target.append([float(i) for i in dt[1].split(' ')])

    return np.array(train), np.array(train_target)

def open_data():
    train, train_target = read_data_from_file("data_lab2/ballist.dat")
    test, test_target = read_data_from_file("data_lab2/balltest.dat")
    return (train, train_target, test, test_target)


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

def weights_init(x):
    weights = np.random.rand(len(x))
    return weights

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
        k = phi_function(xi, mu[i], sigma)
        phi_vector[i] = k
    return phi_vector

def error_function(target, phi_vector, weights):
    error= (target - np.matmul(phi_vector,np.transpose(weights)))
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
        for y in range(net.shape[1]):
            w = net[x][y]
            # don't bother with actual Euclidean distance, to avoid expensive sqrt operation
            sq_dist = np.sum(np.square(w - t))
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

def cl_for_mu_placement(epochs, train, net):
    init_radius = 5
    init_learning_rate = 0.2
    time_constant = epochs / np.log(init_radius)

    for i in range(epochs):
        r = decay_radius(init_radius, i, time_constant)
        l = decay_learning_rate(init_learning_rate, i, epochs)
        for j in range(train.shape[0]):
            # datapoint = np.array([float(i) for i in train[j]])
            bmu, bmu_idx = find_bmu(train[j], net)
            for x in range(net.shape[0]): # number of nodes
                w = net[x]
                w_dist = np.sum((x - bmu_idx) ** 2)
                if w_dist <= r**2:
                    influence = calculate_influence(w_dist, r)
                    new_w = w + (l * influence * (train[j] - w))
                    net[x] = new_w[0]

    return net

def vanillacl_for_mu_placement(epochs, train, net):
    init_learning_rate = 0.2
    for i in range(epochs):
        l = decay_learning_rate(init_learning_rate, i, epochs)
        for j in range(len(train)):
            bmu, bmu_idx = find_bmu(train[j], net)

            #update only the winner node
            influence = 1
            new_w = net[bmu_idx] + (l * influence * (train[j] - net[bmu_idx]))
            net[bmu_idx] = new_w

    return net

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
    for i, elem in enumerate(chunks):
        mean = np.mean(elem)
        mus[i] = mean
    return mus


def rnd_init_mus(nodes_number, train):
    mus = (np.random.permutation(train))[0:(nodes_number - 1)]
    return mus

def main():
    mutyp = 'cl'   #how nodes are initialized. options: "rnd" = random, "mean" = splitting in chunks and getting the means,
                    # "cl" = competitive learning, "vcl" = vanilla competitive learning
    eta = 0.001
    sigma_value=0.4
    epochs = 1#2000
    nodes = (10,10)

    train, train_target, test, test_target = open_data()

    #errors = []
    #for times in range(10):

    if mutyp == 'rnd':
        net = rnd_init_mus(nodes, train)
        marker = '<'
        legend = 'Random'
    elif mutyp == 'mean':
        net = init_mus(nodes, train)
        marker = '^'
        legend = 'Manual'
    elif mutyp == 'cl':
        #Competitive learning
        netsize = nodes+(2,)
        net = np.random.random((10,10,2))
        net = cl_for_mu_placement(epochs, train, net) #2*np.pi *
        marker = '>'
        legend = 'CL'
    elif mutyp == 'vcl':
        #Vanilla competitive learning
        net = vanillacl_for_mu_placement(epochs, train, np.random.random((nodes, 2))) #2*np.pi *
        marker = 'v'
        legend = 'Vanilla CL'
    else:
        print("WRONG MUTYP")
        exit(1)

    # mu = net.flatten()
    mu = net

    #Delta rule:
    weights = weights_init(mu)

    for i in range(epochs):
        sumerror = 0
        for j in range(len(train)):
            phi = phi_vector(train[j], mu, sigma_value)
            error = error_function(target_1[j], phi, weights)
            deltaW = eta*error*phi
            weights = weights + deltaW
            sumerror += (1/2)*error**2
        error = sumerror/len(train)
    print(error)

    #calculate testerror
    testerror = 0
    for j in range(len(test)):
        phi = phi_vector(test[j], mu, sigma_value)
        error = error_function(test_target_1[j], phi, weights)
        testerror += np.abs(error)
    testerror = testerror / len(test)
    print(testerror)
    #    errors.append(testerror)

    #print(legend, "error:", np.mean(errors))




    # plt.title('Convergence')
    # plt.xlabel('Epochs')
    # plt.ylabel('Error')
    # # plt.xlim((0,1500))
    # plt.ylim((0,2))
    # plt.legend()
    # plt.show()



if __name__ == "__main__":
    main()
