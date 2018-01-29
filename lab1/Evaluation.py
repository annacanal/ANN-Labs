import numpy as np
import matplotlib.pyplot as plt


def mean_sq_error(outputs, targets):
    msq =  np.sum((np.power(np.array(outputs) - np.array(targets),2))) / np.size(outputs)
    return msq

def miscl_ratio(outputs, targets):
    miscl = 0
    for i,x in enumerate(targets):
        if (x != outputs[i]):
            miscl += 1
    ratio = miscl/np.size(targets)
    return ratio

def Plot_learning_curve(ratio):
    return 0