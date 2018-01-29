import random
import numpy as np
import matplotlib.pyplot as plt
import Data_Generation
import Evaluation

def W_init(targets,X):
    W = np.zeros([np.size(targets, 0), np.size(X, 0)])
    W = np.random.normal(0,0.001,W.shape)
    return W

def phi(x):
    phi = 2.0 / (1 + np.exp(-x)) - 1
    return phi

def phi_prime(phi):
    phi_prime = (1+phi)*(1- phi)/2.0
    return phi_prime

def forward_pass_general(X, n_layers, n_nodes):
    weights = [n_layers]
    new_inputs = [n_layers+1]
    new_inputs[0] = X
    for layer in range(n_layers):
        weights[layer] = W_init(n_nodes[layer], new_inputs[layer])
        new_inputs[layer+1] = phi(weights[layer].dot(new_inputs[layer]))
        output = new_inputs[layer+1]
        # Add bias to the new inputs for the next iteration
        bias = np.ones([1, np.size(new_inputs[layer+1], 1)])
        new_inputs[layer + 1] = np.concatenate((new_inputs[layer+1], [bias]), axis=0)
    new_inputs[n_layers] = output
    outputs= new_inputs
    return outputs, weights

def forward_pass(X, n_nodes):
    V = W_init(n_nodes[0], X)
    H = phi(V.dot(X))
    # Add bias to the new inputs for the next iteration
    bias = np.ones([1, np.size(H, 1)])
    H = np.concatenate(H, [bias],axis=0)
    W = W_init(n_nodes[1], H)
    O = phi(W.dot(H))
    return V,W,H,O



def backward_pass_general(outputs, weights, targets, n_layers):
    #layers = len(outputs)-1
    delta = np.zeros(1,n_layers)
    delta[0] = (outputs[n_layers] - targets).dot(phi_prime((outputs[n_layers-1]).dot(weights[n_layers-1])))
    for i in range(1, n_layers):
        delta[i] = ((weights[n_layers-i].T).dot(delta[i-1])).dot(phi_prime((outputs[n_layers-i-1]).dot(weights[n_layers-i-1])))
    return delta


def backward_pass(X,V,W,H,O,T, n_nodes):
    H = phi(V.dot(X))
    deltaO = (O-T).dot(phi_prime(O))
    deltaH= (V.T).dot(deltaO).dot(phi_prime(H))
    return deltaO, deltaH

def weight_update_general(eta, delta, outputs, n_layers, alpha, dW, updateW):
    for layer in range(n_layers):
        dW[layer] = (alpha*dW[layer] ) - ( (delta[layer]).dot(outputs[layer].T)*(1 - alpha))
        updateW[layer] = eta*dW[layer]
    return updateW

def weight_update(eta, deltaO, deltaH,X,H, alpha, deltaW,deltaV):
    updateW= (alpha*deltaW) - (deltaH.dot(X.T)*(1-alpha))
    updateV= (alpha*deltaV) - (deltaO.dot(H.T)*(1-alpha))
    deltaW = eta*updateW
    deltaV = eta*updateV
    return deltaW,deltaV

def backforward_prop():
    epochs = 20
    n_nodes = [3,2]
    n_layers = 2
    eta = 0.001
    alpha= 0.9
    deltaW = 0
    deltaV = 0
    patterns, targets = Data_Generation.generate_linearData()
    X = patterns
    outputs = []
    for i in range(epochs):
        V, W, H, O = forward_pass(X, n_nodes)
        deltaO,deltaH = backward_pass(X,V,W,H,O,targets, n_nodes)
        deltaW,deltaV = weight_update(eta, deltaO,deltaH, X,H,alpha,deltaW,deltaV)
        W= W + deltaW
        V= V+deltaV
        print(Evaluation.mean_sq_error(outputs, targets))


# def autoencoder():        The encoder problem - needs the implementation of backprop.
    #Two layer perceptron: 8 input - 3 nodes - 8 outputs
    #Only one node is active: [-1 -1 -1 1 -1 -1 -1 -1]: 1 = active and -1 = nonactive
    # X = np.array([1, -1, -1, -1, -1, -1, -1, -1]).T
    # np.random.shuffle(X)
    # outputs, weights = forward_pass(X, 2, 3)
    # delta = backward_pass(outputs, weights, targets)