from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import linear_model, datasets, metrics
import data_handling
from sklearn.pipeline import Pipeline
import cv2
import matplotlib.pyplot as plt

from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

from scipy.ndimage import convolve

from sklearn.model_selection import train_test_split




def calculate_error(prediction, test):
    error = 0.0
    for i in range(prediction.shape[0]):
        error = error + sum(np.absolute(prediction[i]-test[i]))/prediction.shape[1]
    error = error/prediction.shape[0]
    return error

def main():


    layers=3
    #Read data
    train,train_targets = data_handling.read_train_dataset()
    test, test_targets = data_handling.read_test_dataset()

    ## Models with different number of hidden nodes: 50,75,100,150
    logistic_150 = linear_model.LogisticRegression()
    rbm1_150 = BernoulliRBM(random_state=0, verbose=True)
    rbm2 = BernoulliRBM(random_state=0, verbose=True)
    rbm3 = BernoulliRBM(random_state=0, verbose=True)

    # Hyper-parameters:
    learning_rate = 0.06
    n_iter = 20
    # More components (hidden nodes) tend to give better prediction performance, but larger fitting time

    #Training:
    rbm1_150.learning_rate = learning_rate
    rbm1_150.n_iter = n_iter
    rbm1_150.n_components = 150
    rbm2.learning_rate = learning_rate
    rbm2.n_iter = n_iter
    rbm2.n_components = 120
    rbm3.learning_rate = learning_rate
    rbm3.n_iter = n_iter
    rbm3.n_components = 90
    logistic_150.C = 6000.0

    if layers==1:
        classifier = Pipeline(steps=[("rbm", rbm1_150), ("logistic", logistic_150)])
    if layers==2:
        classifier= Pipeline(steps=[('rbm1', rbm1_150), ('rbm2', rbm2), ('logistic', logistic_150)])
    if layers==3:
        classifier = Pipeline(steps=[('rbm1', rbm1_150), ('rbm2', rbm2), ('rbm3', rbm3), ('logistic', logistic_150)])

        # Training RBM-Logistic Pipeline
    classifier.fit(train, train_targets)
    logistic_classifier = linear_model.LogisticRegression(C=100.0)
    logistic_classifier.fit(train, train_targets)

    # Evaluation
    print("Evaluation:")
    print("Logistic regression using RBM features with 150 hidden nodes:\n%s\n" % (
        metrics.classification_report(
            test_targets,
            classifier.predict(test))))

    print()
    print("Logistic regression using raw pixel features:\n%s\n" % (
        metrics.classification_report(
            test_targets,
            logistic_classifier.predict(test))))

    # Plotting

    plt.figure(figsize=(10, 10))
    for i, comp in enumerate(rbm1_150.components_):
        plt.subplot(10, 15, i + 1)
        plt.imshow(comp.reshape((28, 28)), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle('150 components extracted by RBM', fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    plt.show()

    plt.figure(figsize=(10, 10))
    for i, comp in enumerate(rbm2.components_):
        plt.subplot(10, 12, i + 1)
        plt.imshow(comp.reshape((10, 15)), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle('120 components extracted by RBM', fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    plt.show()

    plt.figure(figsize=(10, 10))
    for i, comp in enumerate(rbm3.components_):
        plt.subplot(10, 10, i + 1)
        plt.imshow(comp.reshape((10, 12)), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle('90 components extracted by RBM', fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    plt.show()


if __name__ == "__main__":
    main()
