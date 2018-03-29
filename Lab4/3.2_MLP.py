from sklearn.neural_network import MLPClassifier
import numpy as np
import data_handling
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata

def calculate_error(prediction, test):
    error = 0.0
    for i in range(prediction.shape[0]):
        error = error + sum(np.absolute(prediction[i]-test[i]))/prediction.shape[1]
    error = error/prediction.shape[0]
    return error

def main():

    #Read data
    train,train_targets = data_handling.read_train_dataset()
    test, test_targets = data_handling.read_test_dataset()

    learning_rate = 0.06
    n_iter = 20

    mlp = MLPClassifier(hidden_layer_sizes=(150,120,90), max_iter=n_iter, alpha=1e-4,
                        solver='sgd', verbose=10, tol=1e-4, random_state=1,
                        learning_rate_init=learning_rate)

    mlp.fit(train, train_targets)
    print("Training set score: %f" % mlp.score(train, train_targets))
    print("Test set score: %f" % mlp.score(test, test_targets))

    fig, axes = plt.subplots(4, 4)
    # use global min / max to ensure all weights are shown on the same scale
    vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
    for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
        ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
                   vmax=.5 * vmax)
        ax.set_xticks(())
        ax.set_yticks(())

    plt.show()





if __name__ == "__main__":
    main()