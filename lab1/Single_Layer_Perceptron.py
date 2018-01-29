import random
import numpy as np
import matplotlib.pyplot as plt
import Data_Generation
import Evaluation


def perceptron_learning_batch(eta,T, X, W):
    Y = predict(X,W)
    DeltaW = eta * (T-Y).dot((X).T)
    return DeltaW

# def perceptron_rule_batch(X, T, W, nu, epochs):
#     for _ in range(epochs):
#         Y = predict(X,W)
#         delta_W = nu/len(T)*np.dot((T-Y),X.T)
#         W = W+delta_W
#     return W
def perceptron_learning_sequential(eta, X, T, W):
    for i in range(len(T)):
        if(((np.dot(W,X[:,i])>0) and (T[i] < 0)) or ((np.dot(W,X[:,i])<0) and (T[i] > 0))):
            deltaW = eta*T[i]*X[:,i]
            W = W+deltaW
    return W

def delta_rule_batch(eta,T, X, W):
    #DeltaW = -eta * (W.dot(X) - T).dot(X.T)
    DeltaW = - eta * (np.subtract(W.dot(X), T)).dot(X.T)
    return DeltaW
#DeltaW = -eta * (np.substract(W.dot(X[:,i]),T[i])).dot(X[:,i].T)

def delta_rule_sequential(eta, T, X, W):
    for i in range(len(T)):
        xx = X[:,i].reshape(-1,1)
        DeltaW = - eta * (W.dot(X[:,i])- T[i]).dot(xx.T)
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

def training():
    epochs = 20
    eta = 0.0001
    errors = []
    acc=[]
    patterns, targets = Data_Generation.generate_linearData()
    #patterns, targets = Data_Generation.generate_nonlinearData()
    X = patterns
    # append bias
    #np.append(X, np.ones(1,np.size(X,1)))
    W = W_init(X)
    type = 'Delta_batch'
    for i in range(epochs):
        if type == 'Perceptron_batch':
            updateW = perceptron_learning_batch(eta,targets, X,W)
        if type == 'Perceptron_sequential':
            updateW = perceptron_learning_sequential(eta, targets, X, W)
        if type == 'Delta_batch':
            updateW = delta_rule_batch(eta, targets, X, W)
        if type == 'Delta_sequential':
            updateW = delta_rule_sequential(eta, targets, X, W)
        W = W + updateW
        error = Evaluation.miscl_ratio(predict(X,W),targets)
        #print(error)
        errors.append(error)
        acc.append(1-error)
    iterations = np.arange(epochs)
    Evaluation.Plot_learning_curve("Learning/iteration", iterations, acc)
    Evaluation.Plot_error_curve("Error/iteration", iterations, errors)
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


training()
