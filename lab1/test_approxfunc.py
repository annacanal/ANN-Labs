import random
import numpy as np
import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D


x = np.arange(-5, 5, 0.5)
y = np.arange(-5, 5, 0.5)
z = np.exp((x*x.T+y*y.T)/10) - 0.5
xv, yv = np.meshgrid(x, y)
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(xv,yv,z)
plt.show()