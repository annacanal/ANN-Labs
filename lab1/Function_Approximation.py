import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from Two_Layer_Func_Approx import backforward_prop
import Two_Layer_Perceptron
import Evaluation
import math

def approx_function():
    X = np.arange(-5, 5, 0.5).T
    Y = np.arange(-5, 5, 0.5).T
    xx, yy = np.meshgrid(X,Y)
    zz = np.exp(-(xx *xx*0.1))*np.exp(-(yy *yy * 0.1)) - 0.5
    N = X.shape[0]*Y.shape[0]
    bias = np.ones((1,N))
    patterns = np.vstack((np.reshape(xx,(1, N)), np.reshape(yy,(1, N)),bias))
    targets = np.reshape(zz,(1, N))
    n_nodes = [3, 2]
    # Visualising data:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # print(zz.shape)
    surf = ax.plot_surface(xx, yy, zz,cmap=plt.cm.BuGn, linewidth=0 )
    plt.title('Function to approximate')
    plt.show()

    plt.figure()
    plt.title('Function to approximate')
    plt.plot(zz)
    plt.show()

    return patterns, targets, n_nodes, xx, yy, X

def average_output(output, n_nodes):
    output_averaged = 0
    for i in range(len(n_nodes)):
        output_averaged = output_averaged + output[i,:]
    output_averaged = output_averaged/len(n_nodes)
    return output_averaged

def backforward_prop(patterns_train, targets_train, n_nodes, xx_train, yy_train, X):
    epochs = 10000
    eta = 0.1
    alpha = 0.9
    deltaW = 0
    deltaV = 0
    # x_test = np.arange(-5, 5, 0.5).T
    # y_test = x_test
    # X_test, Y_test= np.meshgrid(x_test, y_test)
    # N_test= X_test.shape[0]**2
    # bias = np.ones((1,N_test))
    # patterns_test = np.vstack((np.reshape(X_test, (1, N_test)), np.reshape(Y_test, (1, N_test)), bias))

    W = Two_Layer_Perceptron.W_init(n_nodes[0], np.size(patterns_train, 0))
    V = Two_Layer_Perceptron.W_init(n_nodes[1], n_nodes[0] + 1)
    errors_miscl=[]
    errors_mse=[]
    errors_mse_test=[]
    errors_miscl_test=[]
    errors_miscl = []
    errors_mse = []

    for i in range(epochs):
        H, O = Two_Layer_Perceptron.forward_pass(patterns_train, W, V)
        deltaO, deltaH = Two_Layer_Perceptron.backward_pass(patterns_train, W, V, H, O, targets_train, n_nodes)
        deltaW, deltaV = Two_Layer_Perceptron.weight_update(eta, deltaO, deltaH, patterns_train, H, alpha, deltaW, deltaV)
        W = W + deltaW
        V = V + deltaV
        # error_miscl = Evaluation.miscl_ratio(Two_Layer_Perceptron.predict(O, n_nodes), targets_train)
        # errors_miscl.append(error_miscl)
        output = average_output(O, n_nodes)
        error_mse = Evaluation.mean_sq_error(output, targets_train)
        errors_mse.append(error_mse)
    # Extra for function approximation:
    gridsize = X.shape[0]
    output = average_output(O, n_nodes)
    # print(output)
    zz = np.reshape(output,(gridsize, gridsize))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xx_train, yy_train, zz, cmap=plt.cm.BuGn, linewidth=0)
    plt.figure()
    plt.plot(zz)
    plt.show()

    iterations = np.arange(epochs)
    plt.figure()
    Evaluation.Plot_error_curve("MSE/iteration in learning", iterations, errors_mse)
    # Evaluation.Plot_error_curve("Missclassification/iteration in learning", iterations, errors_miscl)

def main():
    patterns, targets, n_nodes, xx, yy, X = approx_function()
    backforward_prop(patterns, targets, n_nodes, xx, yy, X)


if __name__ == "__main__":
    main()
