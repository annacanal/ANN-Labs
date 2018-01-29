import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Two_Layer_Func_Approx import backforward_prop

def approx_function():
    x = np.arange(-1, 1, 0.5)
    y = np.arange(-1, 1, 0.5)
    N = len(x)*len(y)
    z = np.exp((np.outer(x,x)+np.outer(y,y))/10) - 0.5
    xx, yy = np.meshgrid(x, y)
    # print(len(x))
    #targets and patterns for training:
    targets = z.reshape(1, N)
    targets = targets.flatten()
    bias = np.ones(N)
    patterns = np.concatenate((xx.reshape(1, N), yy.reshape(1, N), [bias]), axis=0)
    n_nodes = [3, 2]
    backforward_prop(patterns, targets, n_nodes)
    #Visualising data: 
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(xx, yy, z)
    # plt.show()

    #training with patterns and targets: 


    #Through forward pass get out with "approx function"s patterns as input:
    # n_nodes = 2 #Try for different number of nodes
    # W = W_init(n_nodes, patterns)
    # V = W_init(np.ones((2,200)))
    # W,V,H,O = forward_pass(patterns, W, V)
    # zz = O.reshape(len(xx), len(yy))
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(xx, yy, zz)
    # plt.show()

def main():
    approx_function()

if __name__ == "__main__":
    main()
