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
    return H,O

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


def predict(O):
    prediction = np.round(O)
    for i in range(len(prediction)):
        if prediction[0,i]>0:
            prediction[0,i]= 1.0
        else:
            prediction[0,i]= -1.0
    return prediction


def backforward_prop(nodes):
    epochs = 1000
    n_nodes = [nodes,1]
    n_layers = 2
    eta = 0.01
    alpha= 0.9
    deltaW = 0
    deltaV = 0
    #patterns, targets = Data_Generation.generate_linearData(100)
    patterns, targets = Data_Generation.generate_nonlinearData(100)

    patterns_test, targets_test = Data_Generation.generate_nonlinearData(50)
    X = patterns
    X_test = patterns_test
    W = W_init(n_nodes[0], np.size(X, 0))
    V = W_init(n_nodes[1], n_nodes[0]+1)
    errors_miscl=[]
    errors_mse=[]
    errors_mse_test=[]
    errors_miscl_test=[]
    for i in range(epochs):
        H, O = forward_pass(X, W,V)
        hhh, output_test = forward_pass(X_test, W, V)
        deltaO,deltaH = backward_pass(X,W,V,H,O,targets, n_nodes)
        deltaW,deltaV = weight_update(eta, deltaO,deltaH, X,H,alpha,deltaW,deltaV)
        W = W + deltaW
        V = V + deltaV
        # error_miscl = Evaluation.miscl_ratio(predict(O), targets)
        error_mse = Evaluation.mean_sq_error(O, targets)
        # errors_miscl.append(error_miscl)
        errors_mse.append(error_mse)
        error_mse_test = Evaluation.mean_sq_error(output_test, targets_test)
        # error_miscl_test = Evaluation.miscl_ratio(predict(output_test), targets_test)
        errors_mse_test.append(error_mse_test)
        # errors_miscl_test.append(error_miscl_test)
    iterations = np.arange(epochs)

    return errors_mse,errors_mse_test,iterations


def main():
    n_nodes = [2,5,10,25]
    errors_learning =[]
    errors_test =[]

    for i in range(len(n_nodes)):
        error_learning, error_test,iterations =backforward_prop(n_nodes[i])
        errors_learning.append(error_learning)
        errors_test.append(error_test)

    for i in range(len(n_nodes)):
        name ="MSE/iteration in learning"
        plt.title(name)
        plt.plot(iterations, errors_learning[i], label= "Nodes = "+str(n_nodes[i]))
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.legend()
    plt.show()

    for i in range(len(n_nodes)):
        name= "MSE/iteration in test"
        plt.title(name)
        plt.plot(iterations, errors_test[i], label= "Nodes = "+str(n_nodes[i]))
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main()
