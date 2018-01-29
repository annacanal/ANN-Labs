import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Two_Layer_Perceptron import forward_pass

def approx_function():
    x = np.arange(-5, 5, 0.5)
    y = np.arange(-5, 5, 0.5)
    N = len(x)*len(y)
    z = np.exp((np.outer(x,x)+np.outer(y,y))/10) - 0.5
    xx, yy = np.meshgrid(x, y)
    #targets and patterns for training:
    targets = z.reshape(1, N)
    patterns = np.concatenate((xx.reshape(1, N), yy.reshape(1, N)), axis=0)
    #Visualising data: 
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(xx, yy, z)
    # plt.show()

    #Through forward pass get out with pattern as input data: Experiment with different number of nodes
    n_nodes = 3
    V,W,H,O = forward_pass(patterns, n_nodes)
    print(O)
    # zz = out.reshape

def main():
    approx_function()

if __name__ == "__main__":
    main()
