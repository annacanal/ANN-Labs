import random
import numpy as np
import matplotlib.pyplot as plt
import Data_Generation
import Evaluation


def perceptron_learning_batch(eta,T, X, W):
    Y = predict(X,W)
    DeltaW = eta * (T-Y).dot((X).T)
    return DeltaW

def perceptron_learning_sequential(eta, T, X, W):
    for i in range(len(T)):
        deltaW = eta * (T[i] - np.dot(W, X[:, i])) * X[:, i]
        W = W + deltaW
    return W

def delta_rule_batch(eta,T, X, W):
    #DeltaW = -eta * (W.dot(X) - T).dot(X.T)
    DeltaW = - eta * (np.subtract(W.dot(X), T)).dot(X.T)
    return DeltaW

def delta_rule_sequential(eta, T, X, W):
    for i in range(len(T)):
        DeltaW = - eta * (W.dot(X[:,i])- T[i])*X[:,i]
        W = W + DeltaW
    return W

def W_init(X):
    W = np.zeros([1, np.size(X, 0)])
    W = np.random.normal(0,0.001,W.shape)
   # print(W)
    return W

def predict(X,W):
    prediction = np.round(W.dot(X))
    for i in range(len(prediction)):
        if prediction[0,i]>0:
            prediction[0,i]= 1.0
        else:
            prediction[0,i]= -1.0
    return prediction

def training(types):
    epochs = 20
    eta = 0.0001
    errors = []
    acc=[]
    patterns, targets = Data_Generation.generate_linearData(100)
    #patterns, targets = Data_Generation.generate_nonlinearData(100)
    X = patterns
    # append bias
    #np.append(X, np.ones(1,np.size(X,1)))
    W = W_init(X)
    type = types
    for i in range(epochs):
        if type == 'Perceptron_batch':
            updateW = perceptron_learning_batch(eta, targets, X, W)
            W = W + updateW
        if type == 'Perceptron_sequential':
            W = perceptron_learning_sequential(eta, targets, X, W)
        if type == 'Delta_batch':
            updateW = delta_rule_batch(eta, targets, X, W)
            W = W + updateW
        if type == 'Delta_sequential':
            W = delta_rule_sequential(eta, targets, X, W)

        error = Evaluation.miscl_ratio(predict(X,W),targets)
        #print(error)
        errors.append(error)
        acc.append(1-error)
    iterations = np.arange(epochs)
    # Evaluation.Plot_learning_curve("Learning/iteration "+type+' non-linear data', iterations, acc)
    # Evaluation.Plot_error_curve("Error/iteration "+type+' non-linear data', iterations, errors)
    return iterations, acc,errors

        # p = W[0, :2]
        # k = -W[0, np.size(X, 0)-1] / p.dot(p.T)
        # l = np.sqrt(p.dot(p.T))
        # predA = np.where(targets == 1)
        # axarr[i].scatter(patterns[0,predA], patterns[1,predA], marker='x')
        # predB = np.where(targets == -1)
        # axarr[i].scatter(patterns[0,predB], patterns[1,predB], marker='o')
        # x = np.array([p[0], p[0]]).dot(k) + [-p[1], p[1]]/l
        # y = np.array([p[1], p[1]]).dot(k) + [p[0], -p[0]]/l
        # axarr[i].plot(x, y, color='r')
        # #plt.axis()
        # plt.axis([-31, 31, -31, 31]);
    #plt.show()



def main():
    types1 = ['Perceptron_batch','Perceptron_sequential']
    types2 = ['Delta_batch', 'Delta_sequential']
    errors1 =[]
    accs1 =[]
    errors2 =[]
    accs2 =[]
    for i in range(len(types1)):
        iterations, acc, error =training(types1[i])
        accs1.append(acc)
        errors1.append(error)

    for i in range(len(types1)):
        name ="Learning/iteration linear data"
        plt.title(name)
        plt.plot(iterations, accs1[i], label= types1[i])
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.legend()
    plt.show()

    for i in range(len(types1)):
        name= "Error/iteration linear data"
        plt.title(name)
        plt.plot(iterations, errors1[i], label= types1[i])
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.legend()
    plt.show()
    for i in range(len(types2)):
        iterations, acc, error= training(types2[i])
        accs2.append(acc)
        errors2.append(error)

    for i in range(len(types2)):
        name ="Learning/iteration linear data"
        plt.title(name)
        plt.plot(iterations, accs2[i], label= types2[i])
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.legend()
    plt.show()

    for i in range(len(types2)):
        name= "Error/iteration linear data"
        plt.title(name)
        plt.plot(iterations, errors2[i], label= types2[i])
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.legend()
    plt.show()
if __name__ == "__main__":
    main()
