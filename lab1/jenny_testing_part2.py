from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
import Mackey_Glass_Datapoints
import Evaluation

def regression_evaluation(n_nodes, xtrain, ytrain, xtest, ytesttrue, alpha_list):
    error_mse_test = np.zeros(len(alpha_list))
    for i in range(len(alpha_list)):
        nn = MLPRegressor(
        hidden_layer_sizes=(n_nodes,),  activation='logistic', solver='sgd', alpha=alpha_list[i], #batch_size='auto',
        learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=100000, #shuffle=True,
        random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
        early_stopping=True, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        n = nn.fit(xtrain, ytrain)
        # ytrain = nn.predict(xtrain) 
        ytest = nn.predict(xtest)
        # error_mse = Evaluation.mean_sq_error(ytrain, ytesttrue)
        error_mse_test[i] = Evaluation.mean_sq_error(ytest, ytesttrue)
    return error_mse_test

def main():
    i = np.arange(301, 1500, 5)
    datapoints = (Mackey_Glass_Datapoints.MackeyGlass(25, 0.2, 0.1, 1500))[i]

    # #datapoints = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]
    y = datapoints[5::6]
    # print(y.shape)
    x = datapoints[0:5]
    # print(x.shape)
    for k in range(1, int(np.size(datapoints)/6)):
        x = np.vstack((x, datapoints[(k*6):(k*6+5)]))
    #x = np.transpose(x)

    #xtrain = x[:,:30]
    xtrain = x[:10,:]
    # print(xtrain)
    ytrain = y[:10]
    # print(ytrain)
    # print(ytrain)
    #xtest = x[:,31:]
    xtest = x[20:30,:]
    print(xtest)
    ytesttrue = y[30:]
    # print(ytesttrue)

    # print(ytesttrue)

# nn = MLPRegressor(
#     hidden_layer_sizes=(11,1),  activation='logistic', solver='sgd', alpha=0.01, #batch_size='auto',
#     learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=100000, #shuffle=True,
#     random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
#     early_stopping=True, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    n_nodes = np.array([2, 5, 10, 25])
    alpha_list = np.linspace(0.001, 0.1, 10)
    errors_learning = []
    errors_test = []

    for i in range(len(n_nodes)):
        error_test = regression_evaluation(n_nodes[i], xtrain, ytrain, xtest, ytesttrue, alpha_list)
        # errors_learning.append(error_learning)
        errors_test.append(error_test)
    # print('nodes = ', n_nodes[i])
    # print(errors_test)

    # for i in range(len(n_nodes)):
    #     name ="MSE/iteration in learning"
    #     plt.title(name)
    #     plt.plot(iterations, errors_learning[i], label= "Nodes = "+str(n_nodes[i]))
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Error')
    #     plt.legend()
    # plt.show()

    for i in range(len(n_nodes)):
        name= "MSE/iteration in test"
        plt.title(name)
        plt.plot(alpha_list, errors_test[i], label= "Nodes = "+str(n_nodes[i]))
        plt.xlabel('Alphas')
        plt.ylabel('Error_mse')
        plt.legend()
    plt.show()

    # n = nn.fit(xtrain, ytrain)
    # ytest = nn.predict(xtest)

    # # print(Evaluation.mean_sq_error(ytest, ytesttrue))

    # yplot = np.hstack((ytrain,ytest))
    # xplot = np.vstack((xtrain, xtest))

    # predictionplot = []
    # for j,elem in enumerate(xplot):
    #     predictionplot = np.hstack((predictionplot, elem))
    #     predictionplot = np.hstack((predictionplot, yplot[j]))

    # yi = i[5::6]

    # #in order to plot the output
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.plot(i, datapoints)#, s=1, c='b', marker="s", label='real')
    # ax1.plot(i, predictionplot)
    # #ax1.plot(yi,yplot)#, s=10, c='r', marker="o", label='NN Prediction')
    # # plt.show()

if __name__ == "__main__":
    main()

