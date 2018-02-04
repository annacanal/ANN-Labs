from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
import Mackey_Glass_Datapoints
import Evaluation


def plot_timeseries(xtrain, xvalid, xtest, ytrain, yvalid, ytest, i, datapoints):
    yplot = np.hstack((ytrain, yvalid, ytest))
    xplot = np.vstack((xtrain, xvalid, xtest))

    predictionplot = []
    for j, elem in enumerate(xplot):
        predictionplot = np.hstack((predictionplot, elem))
        predictionplot = np.hstack((predictionplot, yplot[j]))

    yi = i[5::6]

    # in order to plot the output
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(i, datapoints)  # , s=1, c='b', marker="s", label='real')
    ax1.plot(i, predictionplot)

    plt.show()


def train_and_test(datapoints, nodes, tol, alpha, noise_variance):


    #datapoints = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]
    y = datapoints[5::6]

    x = datapoints[0:5]
    for k in range(1, int(np.size(datapoints)/6)):
        x = np.vstack((x, datapoints[(k*6):(k*6+5)]))

    xtrain = x[:25,:]
    ytrain = y[:25]
    xvalid = x[25:30, :]
    yvalidtrue = y[25:30]
    xtest = x[30:,:]
    ytesttrue = y[30:]

    nn = MLPRegressor(
        hidden_layer_sizes=nodes,  activation='logistic', solver='sgd',learning_rate='constant', learning_rate_init=0.001, \
        tol=tol, early_stopping=True, max_iter=100000, alpha=alpha, batch_size='auto', power_t=0.5, random_state=9, \
        verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, validation_fraction=0.1, beta_1=0.9, \
        beta_2=0.999, epsilon=1e-08, shuffle=True)

    n = nn.fit(xtrain, ytrain)
    yvalid = nn.predict(xvalid)
    mse = Evaluation.mean_sq_error(yvalid, yvalidtrue)

    #ytest = nn.predict(xtest)
    #plot_timeseries(xtrain, xvalid, xtest, ytrain, yvalid, ytest, i, datapoints)


    print(mse)
    return mse



def main():

    nodes1 = [1,2,3,4,5,6,7,8]
    for k in range(len(nodes1)):
        nodes = [(nodes1[k],1),(nodes1[k],2),(nodes1[k],3),(nodes1[k],4),(nodes1[k],5),(nodes1[k],6),(nodes1[k],7),(nodes1[k],8)]
        #alphas = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
        alpha = 0.001
        #tols = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
        tol = 0.0001
        noise_variances = [0.03, 0.09, 0.18]

        errors_mse =[]

        # train_and_test((4,2), 0.001, 0.001)

        for j in range(len(noise_variances)):
            dataloc = np.arange(301, 1500, 5)
            datapoints = (Mackey_Glass_Datapoints.MackeyGlass(25, 0.2, 0.1, 1500, noise_variances[j]))[dataloc]
            errors = []
        #   error = train_and_test((4.2), tols[i], 0.001)
            for i in range(len(nodes)):
              error = train_and_test(datapoints, (nodes[i]), tol, alpha, noise_variances[j])
              errors.append(error)
            errors_mse = np.hstack((errors_mse, errors))

        errors_mse = np.array(errors_mse).reshape((len(noise_variances), len(nodes)))
        errors_mse = errors_mse.T

        for i in range(len(nodes)):
            name ="MSE wrt Noise_Variance (1st Layer: "+str(nodes[i][0])+" nodes)"
            plt.title(name)
            plt.ylim(0,0.4)
            plt.plot(noise_variances, errors_mse[i], label= "Nodes: 5-"+str(nodes[i][0])+"-"+str(nodes[i][1])+"-1")
            plt.xlabel('Noise Variannce')
            plt.ylabel('Error')
            plt.legend(loc = 'upper right')
        plt.savefig('\\'+name+'.png')  # save the figure to file
        plt.cla()
        plt.clf()
        # plt.show()

main()