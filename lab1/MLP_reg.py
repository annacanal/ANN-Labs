from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
import Mackey_Glass_Datapoints

x = np.arange(0, 1200, 1).reshape(-1,1)
y = Mackey_Glass_Datapoints.MackeyGlass(25, 0.2, 0.1, 1200)
test_x = x[-200:]
test_x = test_x.reshape(-1, 1)

nn = MLPRegressor(
    hidden_layer_sizes=(5,),  activation='logistic', solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
    random_state=9, tol=0.0001, momentum=0.9, nesterovs_momentum=True,
    early_stopping=True, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

n = nn.fit(x, y)
test_y = nn.predict(test_x)
poang = nn.score(test_x, test_y, sample_weight=None)
print(poang)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(x, y, s=1, c='b', marker="s", label='real')
ax1.scatter(test_x, test_y, s=10, c='r', marker="o", label='NN Prediction')
# plt.show()