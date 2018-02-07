import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Two_Layer_Perceptron
import Evaluation
import math

def approx_function():
    X = np.arange(-5, 5, 0.5).T #20 values
    Y = np.arange(-5, 5, 0.5).T
    xx, yy = np.meshgrid(X,Y)
    zz = np.exp(-(xx *xx*0.1))*np.exp(-(yy *yy * 0.1)) - 0.5    #shape (20,20)
    N = X.shape[0]*Y.shape[0] # = 400


    bias = np.ones((1,N))
    patterns = np.vstack((np.reshape(xx,(1, N)), np.reshape(yy,(1, N)),bias))
    targets = np.reshape(zz,(1, N))
    # Visualising data:
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(xx, yy, zz,cmap=plt.cm.ocean, linewidth=0 )
    # plt.title('Function to approximate')
    # plt.show()


    return patterns,targets,xx,yy,X

#Why training on small amount of data 25, and test on 375? Shouldn't it be the other way around?

def split_train_test(patterns,targets, n_split):
    patterns_train = patterns[:,:n_split]                   #shape is (3,25)
    patterns_test= patterns[:,n_split:patterns.shape[1]]    #shape is (3,375)
    targets_train = targets[:,:n_split]                     #shape is (1,25)
    targets_test = targets[:,n_split:patterns.shape[1]]     #shape is (1,375)
    return patterns_train, patterns_test, targets_train, targets_test

def backforward_prop(patterns, targets, nodes,xx_train,yy_train,X):
    epochs = 1000
    eta = 0.1
    alpha = 0.9
    deltaW = 0
    deltaV = 0
    N_split = 25 #from 1 to 25
    n_nodes = [nodes,1]

    #For plot the approximated function
    # patterns_train=patterns
    # targets_train=targets

    #Suffhle patterns and targets
    s = np.random.permutation(patterns.shape[1])
    patterns = patterns[:,s]
    targets = targets[:,s]
    #Split between train and test
    patterns_train, patterns_test, targets_train, targets_test = split_train_test(patterns,targets, N_split)
    W = Two_Layer_Perceptron.W_init(n_nodes[0], np.size(patterns_train, 0))
    V = Two_Layer_Perceptron.W_init(n_nodes[1], n_nodes[0] + 1)
    errors_mse_test=[]
    errors_mse = []

    for i in range(epochs):
        H, O = Two_Layer_Perceptron.forward_pass(patterns_train, W, V)
        hh, O_test = Two_Layer_Perceptron.forward_pass(patterns_test, W, V)
        deltaO, deltaH = Two_Layer_Perceptron.backward_pass(patterns_train, W, V, H, O, targets_train, n_nodes)
        deltaW, deltaV = Two_Layer_Perceptron.weight_update(eta, deltaO, deltaH, patterns_train, H, alpha, deltaW, deltaV)
        W = W + deltaW
        V = V + deltaV

        error_mse = Evaluation.mean_sq_error(O, targets_train)
        errors_mse.append(error_mse)
        error_mse_test = Evaluation.mean_sq_error(O_test, targets_test)
        errors_mse_test.append(error_mse_test)
    # Extra for function approximation:
    # gridsize = X.shape[0]
    # #output = average_output(O, n_nodes)
    # zz = np.reshape(O,(gridsize, gridsize))
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(xx_train, yy_train, zz, cmap=plt.cm.ocean, linewidth=0)
    # plt.title('Learned Function with ' + str(epochs) + ' epochs and ' + str(n_nodes[0]) + ' nodes')
    # plt.show()
    iterations = np.arange(epochs)
    return errors_mse,errors_mse_test,iterations,N_split
    # plt.figure()
    # Evaluation.Plot_error_curve("MSE/iteration in learning with "+ str(n_nodes[0]) + ' nodes and n= '+str(N_split), iterations, errors_mse)
    # Evaluation.Plot_error_curve("MSE/iteration in test "+ str(n_nodes[0]) + ' nodes and n= '+str(N_split), iterations, errors_mse_test)

def main():
    patterns, targets, xx,yy,X= approx_function()
    n_nodes = [2,5,10,25]
    errors_learning =[]
    errors_test =[]

    for i in range(len(n_nodes)):
        error_learning, error_test, iterations,N_split = backforward_prop(patterns,targets,n_nodes[i],xx,yy,X)
        errors_learning.append(error_learning)
        errors_test.append(error_test)

    for i in range(len(n_nodes)):
        name = "MSE/iteration in learning with n= " +str(N_split)
        plt.title(name)
        plt.plot(iterations, errors_learning[i], label="Nodes = " + str(n_nodes[i]))
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.legend()
    plt.show()

    for i in range(len(n_nodes)):
        name = "MSE/iteration in test with n= " +str(N_split)
        plt.title(name)
        plt.plot(iterations, errors_test[i], label="Nodes = " + str(n_nodes[i]))
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
