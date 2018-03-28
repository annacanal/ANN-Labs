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


def train_classifier(rbm1, logistic,train,train_targets, learning_rate,n_iter,n_hnodes,layers, rbm2,rbm3):
    #rbm = BernoulliRBM(batch_size=10, learning_rate=learning_rate, n_components=n_components, n_iter=n_iter,random_state=None, verbose=0)
    rbm1.learning_rate = learning_rate
    rbm1.n_iter=n_iter
    rbm1.n_components =n_hnodes
    rbm2.learning_rate = learning_rate
    rbm2.n_iter = n_iter
    rbm2.n_components = 100
    logistic.C = 6000.0
    if layers==1:
        classifier = Pipeline(steps=[("rbm", rbm1), ("logistic", logistic)])
    if layers==2:
        classifier= Pipeline(steps=[('rbm1', rbm1), ('rbm2', rbm2), ('logistic', logistic)])
    if layers==3:
        classifier = Pipeline(steps=[('rbm1', rbm1), ('rbm2', rbm2), ('rbm3', rbm3), ('logistic', logistic)])

    # Training RBM-Logistic Pipeline
    classifier.fit(train, train_targets)

    return classifier, rbm1,rbm2



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

    ## Models with different number of hidden nodes: 50,75,100,150
    logistic_150 = linear_model.LogisticRegression()
    rbm_150 = BernoulliRBM(random_state=0, verbose=True)
    rbm2 = BernoulliRBM(random_state=0, verbose=True)
    rbm3 = BernoulliRBM(random_state=0, verbose=True)

    # Hyper-parameters:
    learning_rate = 0.06
    n_iter = 20
    # More components (hidden nodes) tend to give better prediction performance, but larger fitting time

    #Training:
    classifier_150, rbm_150, rbm2_150= train_classifier(rbm_150, logistic_150,train,train_targets, learning_rate,n_iter,n_hnodes=150,layers=1)
    classifier_150, rbm_150, rbm2_150 = train_classifier(rbm_150, logistic_150, train, train_targets, learning_rate, n_iter, n_hnodes=150,layers=2,rbm2=rbm2)
    classifier_150, rbm_150, rbm2_150 = train_classifier(rbm_150, logistic_150, train, train_targets, learning_rate,
                                                         n_iter, n_hnodes=150, layers=2, rbm2=rbm2, rbm3=rbm3)

    logistic_classifier = linear_model.LogisticRegression(C=100.0)
    logistic_classifier.fit(train, train_targets)

    # Evaluation
    print("Evaluation:")
    print("Logistic regression using RBM features with 150 hidden nodes:\n%s\n" % (
        metrics.classification_report(
            test_targets,
            classifier_150.predict(test))))

    print()
    print("Logistic regression using raw pixel features:\n%s\n" % (
        metrics.classification_report(
            test_targets,
            logistic_classifier.predict(test))))

    # Predict test set
    # image from each digit
    example_digits_indexs = [18, 3, 7, 0, 2, 1, 14, 8, 6, 5]  # indexs in the test partition digits: 0,1,2,3,4,5,6,7,8,9
    prediction_150 = rbm_150.gibbs(test).astype(int)
    print(calculate_error(prediction_150, test))




if __name__ == "__main__":
    main()