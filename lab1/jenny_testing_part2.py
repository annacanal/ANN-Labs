from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
import Mackey_Glass_Datapoints
import Evaluation

def regression_evaluation(n_nodes, xtrain, ytrain, xtest, ytesttrue, xval, yval, alpha_list):
    error_mse_test = np.zeros(len(alpha_list))
    for i in range(len(alpha_list)):
        nn = MLPRegressor(
        hidden_layer_sizes=(n_nodes,),  activation='logistic', solver='sgd', alpha=alpha_list[i], #batch_size='auto',
        learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=100000, #shuffle=True,
        random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
        early_stopping=True, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        n = nn.fit(xtrain, ytrain)
        # ytrain = nn.predict(xtrain) 
        yval_predicted = n.predict(xval)
        # error_mse = Evaluation.mean_sq_error(ytrain, ytesttrue)
        error_mse_test[i] = Evaluation.mean_sq_error(yval_predicted, yval)
    return error_mse_test

def main():
    i = np.arange(301, 1500, 5)
    datapoints = (Mackey_Glass_Datapoints.MackeyGlass(25, 0.2, 0.1, 1500))[i]
    y = datapoints[5::6]
    x = datapoints[0:5]
    for k in range(1, int(np.size(datapoints)/6)):
        x = np.vstack((x, datapoints[(k*6):(k*6+5)]))

    xtrain = x[:24,:]
    # print(xtrain, 'this is xtrain')
    ytrain = y[:24]
    # print(ytrain)
    xval = x[24:32:]
    # print(xval, 'this is xval')
    yval = y[24:32]
    # print(ytrain)
    xtest = x[32:40,:]
    # print(xtest, 'this is xtest')
    ytesttrue = y[32:]
    # print(ytesttrue)

    n_nodes = np.array([2, 3, 5, 6, 7])
    best_node = 2
    alpha_list = np.linspace(0.001, 0.1, 10)
    errors_val = []

    for i in range(len(n_nodes)):
        error_val = regression_evaluation(n_nodes[i], xtrain, ytrain, xtest, ytesttrue, xval, yval, alpha_list)
        errors_val.append(error_val)

    error_test = regression_evaluation(best_node, xtrain, ytrain, xtest, ytesttrue, xval, yval, alpha_list)

    for i in range(len(n_nodes)):
        name= "MSE/iteration in test"
        plt.title(name)
        plt.plot(alpha_list, errors_test[i], label= "Nodes = "+str(n_nodes[i]))
        plt.xlabel('Alphas')
        plt.ylabel('Error_mse')
        plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

