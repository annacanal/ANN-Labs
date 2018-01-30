from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
import Mackey_Glass_Datapoints
import Evaluation

i = np.arange(301, 1500, 5)
datapoints = (Mackey_Glass_Datapoints.MackeyGlass(25, 0.2, 0.1, 1500))[i]

#datapoints = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]
y = datapoints[5::6]

x = datapoints[0:5]
for k in range(1, int(np.size(datapoints)/6)):
    x = np.vstack((x, datapoints[(k*6):(k*6+5)]))
#x = np.transpose(x)


#xtrain = x[:,:30]
xtrain = x[:30,:]
ytrain = y[:30]
#xtest = x[:,31:]
xtest = x[30:,:]
ytesttrue = y[30:]

nn = MLPRegressor(
    hidden_layer_sizes=(5,1),  activation='logistic', solver='sgd',# alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.1, power_t=0.5, max_iter=10000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

n = nn.fit(xtrain, ytrain)
ytest = nn.predict(xtest)
print(Evaluation.mean_sq_error(ytest, ytesttrue))


yplot = np.hstack((ytrain,ytest))
yi = i[5::6]

#in order to plot the output
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(i, datapoints)#, s=1, c='b', marker="s", label='real')
ax1.plot(yi,yplot)#, s=10, c='r', marker="o", label='NN Prediction')
plt.show()