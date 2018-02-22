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



def weights_init(x):
    weights = np.random.rand(x.shape[0])
    return weights

def sin_function(x):
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = np.sin(2*x[i])
    return y

def phi_function(x, mu, sigma):
    k = x-mu
    phi = np.exp(-(x-mu)**2/(2*sigma**2))
    return phi

def phi_vector(xi, mu, sigma):
    phi_vector = np.zeros((mu.shape[0], mu.shape[1]))
    for i in range(mu.shape[0]):
        phi_vector[i] = phi_function(xi, mu[i], sigma)
    return phi_vector

def error_function(target, phi_vector, weights):
    error= target - np.dot(weights,phi_vector)
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

def cl_for_mu_placement(epochs, train, net):
    init_radius = 70
    init_learning_rate = 0.2
    time_constant = epochs / np.log(init_radius)

    for i in range(epochs):
        r = decay_radius(init_radius, i, time_constant)
        l = decay_learning_rate(init_learning_rate, i, epochs)
        for j in range(len(train)):
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
    sigma_value=0.3
    epochs = 1000
    nodes = 10

    train, train_target, test, test_target = open_data()

<<<<<<< HEAD
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
        net = np.random.random((nodes, train.shape[1]))
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

    #plt.scatter(net.shape[1])

    # mu = net.flatten()
    mu = net                #10,2

    #Delta rule:
    weights = weights_init(mu)
    phi_vecs=[]
    prediction=[]
    for i in range(epochs):
        sumerror = 0
        for j in range(train.shape[0]):
            phi = phi_vector(train[j], mu, sigma_value)
            error = error_function(train_target[j], phi, weights)
            deltaW = eta*np.dot(error,np.transpose(phi))
            weights = weights + deltaW
            sumerror += (1/2)*error**2
            phi_vecs.append(phi)
            prediction.append(np.dot(weights,phi))
        error = sumerror/len(train)
    print(error)

    #calculate testerror
    testerror = 0
    phi_vecs_test=[]
    prediction_test = []
    for j in range(len(test)):
        phi_test = phi_vector(test[j], mu, sigma_value)
        error = error_function(test_target[j], phi, weights)
        testerror += np.abs(error)
        phi_vecs_test.append(phi_test)
        prediction_test.append(np.dot( weights,phi_test))
    testerror = testerror / len(test)
    # print(testerror)
    #    errors.append(testerror)

    #print(legend, "error:", np.mean(errors))
=======
        #errors = []
        #for times in range(10):

    for nodes in [10]:

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
            net = np.random.random((nodes, train.shape[1]))
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

        plt.title('Node placements')
        a = net[:,0]
        b = net[:,1]
        plt.scatter(a, b, label = 'nodes')
        plt.scatter(train[:,0], train[:,1], label = 'train input')
        plt.legend(loc = 'upper left')
        plt.show()
        exit(0)

        # mu = net.flatten()
        mu = net

        #Delta rule:
        weights = weights_init(mu)

        for i in range(epochs):
            sumerror = 0
            for j in range(train.shape[0]):
                phi = phi_vector(train[j], mu, sigma_value)
                error = error_function(train_target[j], phi, weights)
                deltaW = eta*np.dot(error,np.transpose(phi))
                weights = weights + deltaW
                sumerror += (1/2)*error**2
            error = sumerror/len(train)
        print('Nodes: ', nodes,'Train error: ',error)

        #calculate testerror
        testerror = 0
        for j in range(len(test)):
            phi = phi_vector(test[j], mu, sigma_value)
            error = error_function(test_target[j], phi, weights)
            testerror += np.abs(error)
        testerror = testerror / len(test)
        print('Nodes: ', nodes,'Train error: ', testerror)

        print('--------')
        #    errors.append(testerror)

        #print(legend, "error:", np.mean(errors))

>>>>>>> 4fb4765d67556f4eb649b96e0b97e3f8e969478a

    #train prediction
    #prediction = np.dot(phi_vecs,weights)

    prediction1=[]
    prediction2=[]
    prediction_test1=[]
    prediction_test2=[]
    train_target1=[]
    train_target2=[]
    test_target1=[]
    test_target2=[]
    for i in range(len(prediction)):
        prediction1.append(prediction[i][0])
        prediction2.append(prediction[i][1])
    for i in range(len(prediction_test)):
        prediction_test1.append(prediction_test[i][0])
        prediction_test2.append(prediction_test[i][1])

    for i in range(len(train_target)):
        train_target1.append(train_target[i][0])
        train_target2.append(train_target[i][1])
    for i in range(len(test_target)):
        test_target1.append(test_target[i][0])
        test_target2.append(test_target[i][1])


    name = "train approximation, CL with nodes = "+str(nodes)
    plt.title(name)
    plt.scatter(prediction1,prediction2,s=2.5, label="Prediction")
<<<<<<< HEAD
    
=======
>>>>>>> 4fb4765d67556f4eb649b96e0b97e3f8e969478a
    plt.scatter(train_target1,train_target2, s=2.5, label="Target")
    plt.legend()
    plt.show()
    #test prediction
    name = "test approximation, CL with nodes = " + str(nodes)
    plt.title(name)
    plt.scatter(prediction_test1,prediction_test2, s=2.5, label="Prediction")
    plt.scatter(test_target1,test_target2, s=2.5, label="Target")
    plt.legend()
    plt.show()

    # plt.title('Convergence')
    # plt.xlabel('Epochs')
    # plt.ylabel('Error')
    # # plt.xlim((0,1500))
    # plt.ylim((0,2))
    # plt.legend()
    # plt.show()



if __name__ == "__main__":
    main()