import numpy as np
import matplotlib.pyplot as plt


def mean_sq_error(outputs, targets):
    msq =  np.sum((np.power(np.array(outputs) - np.array(targets),2))) / np.size(outputs)
    return msq

def miscl_ratio(outputs, targets):
    miscl = 0
    for i,x in enumerate(targets):
        if (x != outputs[0,i]):
            miscl += 1
    ratio = miscl/np.size(targets)
    return ratio

#Plot the error in each iteration.
def Plot_learning_curve(name,iterations,errors):
    plt.title(name)
    plt.plot(iterations, errors,'blue')
    plt.show()
