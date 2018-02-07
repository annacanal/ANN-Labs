from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
import Mackey_Glass_Datapoints
import Evaluation

def regression_evaluation(n_nodes, xtrain, ytrain, x, alpha_list):
    nn = MLPRegressor(
    hidden_layer_sizes=(n_nodes,),  activation='logistic', solver='sgd', alpha=alpha_list, #batch_size='auto',
    learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=100000, #shuffle=True,
    random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=True, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    n = nn.fit(xtrain, ytrain)
    y = n.predict(x)
    return y

def main():
    i = np.arange(301, 1500, 5)
    datapoints = (Mackey_Glass_Datapoints.MackeyGlass(25, 0.2, 0.1, 1500))[i]
    y = datapoints[5::6]
    x = datapoints[0:5]
    for k in range(1, int(np.size(datapoints)/6)):
        x = np.vstack((x, datapoints[(k*6):(k*6+5)]))

    xtrain = x[:25,:]
    # print(xtrain, 'this is xtrain')
    ytrain = y[:25]
    # print(ytrain)
    xval = x[25:30:]
    # print(xval, 'this is xval')
    yval = y[25:30]
    # print(ytrain)
    xtest = x[30:40,:]
    # print(xtest, 'this is xtest')
    ytesttrue = y[30:]
    # print(ytesttrue)

    n_nodes = np.array([2, 3, 5, 6, 7])
    alpha_list = np.linspace(0.001, 0.1, 5)
    errors_val = []

# Task 4.3.1_3 (Question 3) - training and then decide best model from validation set: 
    # for i in range(len(n_nodes)):
    #     error_val = np.zeros(len(alpha_list))
    #     for j in range(len(alpha_list)):
    #         y_predicted = regression_evaluation(n_nodes[i], xtrain, ytrain, xval, alpha_list[j])
    #         error_val[j] = Evaluation.mean_sq_error(y_predicted, yval)
    #     errors_val.append(error_val)

    # for i in range(len(n_nodes)):
    #     name= "Validation set"
    #     plt.title(name)
    #     plt.plot(alpha_list, errors_val[i], label= "Nodes = "+str(n_nodes[i]))
    #     plt.xlabel('Alphas')
    #     plt.ylabel('Error_mse')
    #     plt.legend()
    # plt.show()

# Task 4.3.1_4 (Question 4) - Choosing best model and plot: 
    best_node = 2
    best_alpha = 0.001
    y_predicted = regression_evaluation(best_node, xtrain, ytrain, xtest, best_alpha)
    
    


    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.plot(i, datapoints)#, s=1, c='b', marker="s", label='real')
    # ax1.plot(i, predictionplot)
    # #ax1.plot(yi,yplot)#, s=10, c='r', marker="o", label='NN Prediction')
    # plt.show()

if __name__ == "__main__":
    main()

