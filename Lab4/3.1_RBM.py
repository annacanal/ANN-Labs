from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
import numpy as np
from sklearn import linear_model, datasets, metrics
import data_handling
from sklearn.pipeline import Pipeline
import cv2
import matplotlib as plt


def nudge(X, y):
    # initialize the translations to shift the image one pixel
    # up, down, left, and right, then initialize the new data
    # matrix and targets
    translations = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    data = []
    target = []

    # loop over each of the digits
    for (image, label) in zip(X, y):
        # reshape the image from a feature vector of 784 raw
        # pixel intensities to a 28x28 'image'
        image = image.reshape(28, 28)

        # loop over the translations
        for (tX, tY) in translations:
            # translate the image
            M = np.float32([[1, 0, tX], [0, 1, tY]])
            trans = cv2.warpAffine(image, M, (28, 28))

            # update the list of data and target
            data.append(trans.flatten())
            target.append(label)

    # return a tuple of the data matrix and targets
    return (np.array(data), np.array(target))


def train_classifier(train,train_targets, learning_rate,n_iter,n_components):
    rbm = BernoulliRBM(batch_size=10, learning_rate=learning_rate, n_components=n_components, n_iter=n_iter,
                 random_state=None, verbose=0)
    logistic = LogisticRegression()
    classifier = Pipeline([("rbm", rbm), ("logistic", logistic)])

   # rbm.intercept_hidden_=0
    #rbm.intercept_visible_=0
    logistic.C = 6000.0

    # Training RBM-Logistic Pipeline
    classifier.fit(train, train_targets)

    # Training Logistic regression
    logistic_classifier = linear_model.LogisticRegression(C=100.0)
    logistic_classifier.fit(train, train_targets)

    return classifier, logistic


def main():
    #Read data
    train,train_targets = data_handling.read_train_dataset()
    test, test_targets = data_handling.read_test_dataset()

    # Hyper-parameters. These were set by cross-validation,
    # using a GridSearchCV. Here we are not performing cross-validation to
    # save time.
    learning_rate = 0.06
    n_iter =20# [10, 20]
    # More components tend to give better prediction performance, but larger
    # fitting time
    n_components = 100

    print("train shape:")
    print(train.shape)
    #print(train_targets.shape)
    #Training
    rbm_clf, logistic_clf = train_classifier(train,train_targets, learning_rate,n_iter,n_components)

    # Evaluation

    print()
    print("Logistic regression using RBM features:\n%s\n" % (
        metrics.classification_report(
            test_targets,
            rbm_clf.predict(test))))

    # print("Logistic regression using raw pixel features:\n%s\n" % (
    #     metrics.classification_report(
    #         test_targets,
    #         logistic_clf.predict(test))))

    # Plotting

    # plt.figure(figsize=(4.2, 4))
    # for i, comp in enumerate(rbm_clf.components_):
    #     plt.subplot(10, 10, i + 1)
    #     plt.imshow(comp.reshape((28, 28)), cmap=plt.cm.gray_r,
    #                interpolation='nearest')
    #     plt.xticks(())
    #     plt.yticks(())
    # plt.suptitle('100 components extracted by RBM', fontsize=16)
    # plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    #
    # plt.show()

if __name__ == "__main__":
    main()
