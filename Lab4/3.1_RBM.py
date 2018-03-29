from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import linear_model, datasets, metrics
import data_handling
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


def train_classifier(rbm, logistic,train,train_targets, learning_rate,n_iter,n_hnodes):
    #rbm = BernoulliRBM(batch_size=10, learning_rate=learning_rate, n_components=n_components, n_iter=n_iter,random_state=None, verbose=0)
    rbm.learning_rate = learning_rate
    rbm.n_iter=n_iter
    rbm.n_components =n_hnodes
    logistic.C = 6000.0
    classifier = Pipeline(steps=[("rbm", rbm), ("logistic", logistic)])

    # Training RBM-Logistic Pipeline
    classifier.fit(train, train_targets)

    return classifier, rbm



def plot_images(rbm_50,rbm_75,rbm_100,rbm_150):
    # 50 nodes
    plt.figure(figsize=(10, 10))
    for i, comp in enumerate(rbm_50.components_):
        plt.subplot(10, 10, i + 1)
        plt.imshow(comp.reshape((28, 28)), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle('50 components extracted by RBM', fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    plt.show()
    # 75 nodes
    # plt.figure(figsize=(10, 10))
    # for i, comp in enumerate(rbm_75.components_):
    #     plt.subplot(10, 10, i + 1)
    #     plt.imshow(comp.reshape((28, 28)), cmap=plt.cm.gray_r,
    #                interpolation='nearest')
    #     plt.xticks(())
    #     plt.yticks(())
    # plt.suptitle('75 components extracted by RBM', fontsize=16)
    # plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    # plt.show()
    # 100 nodes
    plt.figure(figsize=(10, 10))
    for i, comp in enumerate(rbm_100.components_):
        plt.subplot(10, 10, i + 1)
        plt.imshow(comp.reshape((28, 28)), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle('100 components extracted by RBM', fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    plt.show()
    # 150 nodes
    # plt.figure(figsize=(10, 10))
    # for i, comp in enumerate(rbm_150.components_):
    #     plt.subplot(10, 15, i + 1)
    #     plt.imshow(comp.reshape((28, 28)), cmap=plt.cm.gray_r,
    #                interpolation='nearest')
    #     plt.xticks(())
    #     plt.yticks(())
    # plt.suptitle('150 components extracted by RBM', fontsize=16)
    # plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    # plt.show()

    return 0


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
    logistic_50 = linear_model.LogisticRegression()
    logistic_75 = linear_model.LogisticRegression()
    logistic_100 = linear_model.LogisticRegression()
    logistic_150 = linear_model.LogisticRegression()

    rbm_50 = BernoulliRBM(random_state=0, verbose=True)
    rbm_75 = BernoulliRBM(random_state=0, verbose=True)
    rbm_100 = BernoulliRBM(random_state=0, verbose=True)
    rbm_150 = BernoulliRBM(random_state=0, verbose=True)

    # Hyper-parameters:
    learning_rate = 0.06
    n_iter = 20
    # More components (hidden nodes) tend to give better prediction performance, but larger fitting time

    error_50=[]
    error_75=[]
    error_100=[]
    error_150=[]


    #Training:
    classifier_50,rbm_50 = train_classifier(rbm_50, logistic_50,train,train_targets, learning_rate,n_iter,n_hnodes=50)
    classifier_75,rbm_75 = train_classifier(rbm_75, logistic_75,train,train_targets, learning_rate,n_iter,n_hnodes=75)
    classifier_100,rbm_100 = train_classifier(rbm_100, logistic_100,train,train_targets, learning_rate,n_iter,n_hnodes=100)
    classifier_150, rbm_150= train_classifier(rbm_150, logistic_150,train,train_targets, learning_rate,n_iter,n_hnodes=150)



    # Evaluation
    print("Evaluation:")
    print("Logistic regression using RBM features with 50 hidden nodes:\n%s\n" % (
        metrics.classification_report(
            test_targets,
            classifier_50.predict(test))))

    print("Logistic regression using RBM features with 75 hidden nodes:\n%s\n" % (
        metrics.classification_report(
            test_targets,
            classifier_75.predict(test))))
    print("Logistic regression using RBM features with 100 hidden nodes:\n%s\n" % (
        metrics.classification_report(
            test_targets,
            classifier_100.predict(test))))
    print("Logistic regression using RBM features with 150 hidden nodes:\n%s\n" % (
        metrics.classification_report(
            test_targets,
            classifier_150.predict(test))))

    # Plotting
    plot_images(rbm_50,rbm_75,rbm_100,rbm_150)

    # Predict test set
    # image from each digit
    example_digits_indexs = [18, 3, 7, 0, 2, 1, 14, 8, 6, 5]  # indexs in the test partition digits: 0,1,2,3,4,5,6,7,8,9
    prediction_50 = rbm_50.gibbs(test).astype(int)
    prediction_75 = rbm_75.gibbs(test).astype(int)
    prediction_100 = rbm_100.gibbs(test).astype(int)
    prediction_150 = rbm_150.gibbs(test).astype(int)
    print(calculate_error(prediction_50, test))
    print(calculate_error(prediction_75, test))
    print(calculate_error(prediction_100, test))
    print(calculate_error(prediction_150, test))

    plt.figure(figsize=(20, 20))

    for index, i in enumerate(example_digits_indexs):
        plt.subplot(10, 5, 5 * index + 1)
        plt.imshow(test[i].reshape((28, 28)), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.subplot(10, 5, 5 * index + 2)
        plt.imshow(prediction_50[i].reshape((28, 28)), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.subplot(10, 5, 5 * index + 3)
        plt.imshow(prediction_75[i].reshape((28, 28)), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.subplot(10, 5, 5 * index + 4)
        plt.imshow(prediction_100[i].reshape((28, 28)), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.subplot(10, 5, 5 * index + 5)
        plt.imshow(prediction_150[i].reshape((28, 28)), cmap=plt.cm.gray_r,
                   interpolation='nearest')
    plt.show()

    # for i in range(n_iter):
    #     # Training
    #
    #     classifier_50, rbm_50 = train_classifier(rbm_50, logistic_50, train, train_targets, learning_rate, i,n_hnodes=50)
    #     classifier_75, rbm_75 = train_classifier(rbm_75, logistic_75, train, train_targets, learning_rate, i,  n_hnodes=75)
    #     classifier_100, rbm_100 = train_classifier(rbm_100, logistic_100, train, train_targets, learning_rate, i, n_hnodes=100)
    #     classifier_150, rbm_150 = train_classifier(rbm_150, logistic_150, train, train_targets, learning_rate, i, n_hnodes=150)
    # #Predict test set
    # # image from each digit
    #     example_digits_indexs = [18,3,7,0,2,1,14,8,6,5]  # indexs in the test partition digits: 0,1,2,3,4,5,6,7,8,9
    #     prediction_50 = rbm_50.gibbs(test).astype(int)
    #     prediction_75 = rbm_75.gibbs(test).astype(int)
    #     prediction_100 = rbm_100.gibbs(test).astype(int)
    #     prediction_150 = rbm_150.gibbs(test).astype(int)
    #
    #     error_50.append(calculate_error(prediction_50, test))
    #     error_75.append(calculate_error(prediction_75, test))
    #     error_100.append(calculate_error(prediction_100, test))
    #     error_150.append(calculate_error(prediction_150, test))
    #     print("iterations: "+str(i))
    #     print(calculate_error(prediction_50, test))
    #     print(calculate_error(prediction_75, test))
    #     print(calculate_error(prediction_100, test))
    #     print(calculate_error(prediction_150, test))
    #
    # iterations = np.arange(n_iter)
    # plt.plot(iterations, error_50, color='blue',  label= "Nodes = 50")
    # plt.plot(iterations, error_75, color='red', label= "Nodes = 75")
    # plt.plot(iterations, error_100, color='green',label= "Nodes = 100")
    # plt.plot(iterations, error_150, color='orange',label= "Nodes = 150")
    # plt.xlabel('Epochs')
    # plt.ylabel('Mean Error')
    # plt.legend()
    # plt.show()




if __name__ == "__main__":
    main()
