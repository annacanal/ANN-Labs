import random
import numpy as np
import matplotlib.pyplot as plt
import Data_Generation
import Evaluation


def perceptron_learning_batch(T, X, W):
    eta = 0.001
    output = W.dot(X)
    sign = output - T
    sign[sign > 0] = 1
    sign[sign < 0] = -1
    DeltaW = eta * sign.dot(np.array(X).T)
    return DeltaW

def delta_rule_batch(eta,T, X, W):
    DeltaW = -eta * (W.dot(X) - T).dot(X.T)
    return DeltaW

def delta_rule_sequential(eta, T, X, W):
    for i in range(len(T)):
        DeltaW = -eta * (W.dot(X[:,i]) - T[i]).dot(X[:,i].T)
        W = W + DeltaW
    return W

def W_init(targets,X):
    W = np.zeros([np.size(targets, 0), np.size(X, 0)])
    W = np.random.normal(0,0.001,W.shape)
    return W

def training():
    epochs = 20
    eta = 0.001
    patterns, targets = Data_Generation.generate_linearData()
    X = patterns
    # append bias
    #np.append(X, np.ones(1,np.size(X,1)))
    W = W_init(targets,X)

    fig, axarr = plt.subplots(epochs, 1, figsize=(10, 40))
    for i in range(epochs):
        updateW = delta_rule_batch(eta,targets, X,W)
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
        plt.axis([-31, 31, -31, 31]);
    plt.show()