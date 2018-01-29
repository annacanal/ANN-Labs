import random
import numpy as np
import matplotlib.pyplot as plt
import Data_Generation
import Evaluation

def W_init(nodes,input):
    W = np.zeros([nodes, input])
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
        bias = np.ones(np.size(new_inputs[layer+1], 1))
        new_inputs[layer + 1] = np.concatenate((new_inputs[layer+1], [bias]), axis=0)
    new_inputs[n_layers] = output
    outputs= new_inputs
    return outputs, weights

def backward_pass_general(outputs, weights, targets, n_layers):
    #layers = len(outputs)-1
    delta = np.zeros(1,n_layers+1)
    delta[0] = (outputs[n_layers] - targets).dot(phi_prime((outputs[n_layers-1]).dot(weights[n_layers-1])))
    for i in range(1, n_layers):
        delta[i] = ((weights[n_layers-i].T).dot(delta[i-1])).dot(phi_prime((outputs[n_layers-i-1]).dot(weights[n_layers-i-1])))
    return delta


def weight_update_general(eta, delta, outputs, n_layers, alpha, dW, updateW):
    for layer in range(n_layers):
        dW[layer] = (alpha*dW[layer] ) - ( (delta[layer]).dot(outputs[layer].T)*(1 - alpha))
        updateW[layer] = eta*dW[layer]
    return updateW

def forward_pass(X,W,V):
    H = phi(W.dot(X))
    # Add bias to the new inputs for the next iteration
    bias = np.ones(np.size(H, 1))
    H = np.concatenate((H,[bias]),axis=0)
    O = phi(V.dot(H))
    return W,V,H,O

def backward_pass(X,W,V,H,O,T, n_nodes):
    deltaO = (O-T)*(phi_prime(O))
    deltaH = (V.T).dot(deltaO)*(phi_prime(H))
    deltaH = deltaH[:-1,:] #remove bias term
    return deltaO, deltaH

def weight_update(eta, deltaO, deltaH,X,H, alpha, deltaW,deltaV):
    updateW= (alpha*deltaW) - (deltaH.dot(X.T)*(1-alpha))
    updateV= (alpha*deltaV) - (deltaO.dot(H.T)*(1-alpha))
    deltaW = eta*updateW
    deltaV = eta*updateV
    return deltaW,deltaV


def backforward_prop():
    epochs = 100
    n_nodes = [3,2]
    n_layers = 2
    eta = 0.1
    alpha= 0.9
    deltaW = 0
    deltaV = 0

    patterns, targets = Data_Generation.generate_linearData(100)
    #patterns, targets = Data_Generation.generate_nonlinearData(100)

    patterns_test, targets_test = Data_Generation.generate_linearData(50)
    X = patterns
    W = W_init(n_nodes[0], np.size(X, 0))
    V = W_init(n_nodes[1], n_nodes[0]+1)
    print(V.shape)
    errors_miscl=[]
    errors_mse=[]
    for i in range(epochs):
        W, V, H, O = forward_pass(X, W,V)
        deltaO,deltaH = backward_pass(X,W,V,H,O,targets, n_nodes)
        deltaW,deltaV = weight_update(eta, deltaO,deltaH, X,H,alpha,deltaW,deltaV)
        W = W + deltaW
        V = V + deltaV
        error_miscl = Evaluation.miscl_ratio(O, targets)
        error_mse = Evaluation.mean_sq_error(O,targets)
        # print(error)
        #errors_miscl.append()
        errors_mse.append(error_mse)
    iterations = np.arange(epochs)
    Evaluation.Plot_error_curve("Error/iteration", iterations, errors_mse)

# def autoencoder():        The encoder problem - needs the implementation of backprop.
    #Two layer perceptron: 8 input - 3 nodes - 8 outputs
    #Only one node is active: [-1 -1 -1 1 -1 -1 -1 -1]: 1 = active and -1 = nonactive
    # X = np.array([1, -1, -1, -1, -1, -1, -1, -1]).T
    # np.random.shuffle(X)
    # outputs, weights = forward_pass(X, 2, 3)
    # delta = backward_pass(outputs, weights, targets)

backforward_prop()
