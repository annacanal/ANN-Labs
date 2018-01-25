import numpy as np
import matplotlib.pyplot as plt

mean_x = [-20, -20]
mean_y = [20, 20]
cov_x = [[100, 0], [0, 100]]
cov_y = [[100, 0], [0, 100]]
N = 100


x1, x2 = np.random.multivariate_normal(mean_x, cov_x, N).T
y1, y2 = np.random.multivariate_normal(mean_y, cov_y, N).T


plt.plot(x1, x2,  'x')
plt.plot(y1, y2, 'o')
plt.axis('equal')
plt.show()