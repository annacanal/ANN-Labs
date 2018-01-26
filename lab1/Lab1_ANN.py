import random
import numpy as np
import matplotlib.pyplot as plt

def generate_linearData():
    mean_x = [-30, -30]
    mean_y = [30, 30]
    cov_x = [[100, 0], [0, 100]]
    cov_y = [[100, 0], [0, 100]]
    N = 5

    x1, x2 = np.random.multivariate_normal(mean_x, cov_x, N).T
    y1, y2 = np.random.multivariate_normal(mean_y, cov_y, N).T

    classA = np.column_stack((x1, x2)).T
    classB = np.column_stack((y1, y2)).T

    X = np.append(classA[0], classB[0])
    Y = np.append(classA[1], classB[1])
    T = np.append(np.ones(len(classA[0])), -np.ones(len(classB[0])))
    bias = np.ones(2 * N)

    s = np.random.permutation(2 * N)
    patterns = np.concatenate(([X[s]], [Y[s]], [bias]), axis=0)
    target = T[s]

    print(patterns)
    print(target)

    # plt.plot(x1, x2,  'x')
    # plt.plot(y1, y2, 'o')
    # plt.axis('equal')
    # plt.show()
    return patterns, target

def generate_nonlinearData():
    mean_x = [15, 0]
    mean_y = [0, 15]
    cov_x = [[100, 0], [0, 100]]
    cov_y = [[100, 0], [0, 100]]
    N = 100

    x1, x2 = np.random.multivariate_normal(mean_x, cov_x, N).T
    y1, y2 = np.random.multivariate_normal(mean_y, cov_y, N).T

    classA = np.column_stack((x1, x2)).T
    classB = np.column_stack((y1, y2)).T

    X = np.append(classA[0], classB[0])
    Y = np.append(classA[1], classB[1])
    T = np.append(np.ones(len(classA[0])), -np.ones(len(classB[0])))
    bias = np.ones(2 * N)

    s = np.random.permutation(2 * N)
    patterns = np.concatenate(([X[s]], [Y[s]], [bias]), axis=0)
    target = T[s]
    return patterns, target

def delta_rule(T, X, W):
    eta = 0.001
    updateW = -eta * (W.dot(X) - T).dot(X.T)
    return updateW

def W_init(targets,X):
    W = np.zeros([np.size(targets, 0), np.size(X, 0)])
    W = np.random.normal(0,0.001,W.shape)
    return W

def training():
    epochs = 20
    patterns, targets = generate_linearData()
    X = patterns
    # append bias
    #np.append(X, np.ones(1,np.size(X,1)))
    W = W_init(targets,X)

    fig, axarr = plt.subplots(epochs, 1, figsize=(10, 40))
    for i in range(epochs):
        updateW = delta_rule(targets, X,W)
        W = W + updateW
        # output = W.dot(X)
        # predictionA = np.where(targets == 1)
        # predictionB = np.where(targets == -1)
        # plt.figure()
        # plt.plot(X[0,predictionA], X[1,predictionA],  'x', 'red')
        # plt.plot(X[0,predictionB],X[1,predictionB], 'o','blue')
        # plt.contour(output)
        # plt.axis('equal')
        # plt.show()

        p = W[0, :2]
        k = -W[0, np.size(X, 0)-1] / p.dot(p.T)
        l = np.sqrt(p.dot(p.T))
        predA = np.where(targets == 1)
        axarr[i].scatter(patterns[0,predA], patterns[1,predA], marker='x')
        predB = np.where(targets == -1)
        axarr[i].scatter(patterns[0,predB], patterns[1,predB], marker='o')
        x = np.array([p[0], p[0]]).dot(k) + [-p[1], p[1]]/l
        y = np.array([p[1], p[1]]).dot(k) + [p[0], -p[0]]/l
        axarr[i].plot(x, y, color='r')
        #plt.axis()
        plt.show()

def phi(x):
    phi = 2.0 / (1 + np.exp(-x)) - 1
    return phi

def phi_prime(phi):
    phi_prime = (1+phi)*(1- phi)/2.0
    return phi_prime


def forward_pass(X, n_layers, n_nodes):
    weights = np.zeros([1,np.size(n_layers)])
    new_inputs = np.zeros([1, np.size(n_layers)])
    new_inputs[0] = X
    for layer in range(n_layers):
        weights[layer] = W_init(n_nodes[layer], new_inputs[layer])
        new_inputs[layer+1] = phi(weights[layer].dot(new_inputs[layer]))
        output = new_inputs[layer+1]
        # Add bias to the new inputs for the next iteration
        bias = np.ones([1, np.size(new_inputs[layer+1], 1)])
        new_inputs[layer + 1] = np.concatenate((new_inputs[layer+1], [bias]), axis=0)
    new_inputs[n_layers-1] = output
    outputs= new_inputs
    return outputs, weights

def backward_pass():
    return 0


def backforward_prop():
    epochs = 20
    n_nodes = [3,2]
    n_layers = 2
    patterns, targets = generate_linearData()
    X = patterns
    for i in range(epochs):
        outputs,weights = forward_pass(X,n_nodes, n_layers)
        backward_pass()
        updateW = weight_update()
        W = W + updateW
    return 0

training()

