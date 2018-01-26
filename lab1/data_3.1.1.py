import numpy as np
import matplotlib.pyplot as plt

mean_x = [-20, -20]
mean_y = [20, 20]
cov_x = [[100, 0], [0, 100]]
cov_y = [[100, 0], [0, 100]]
N = 5


x1, x2 = np.random.multivariate_normal(mean_x, cov_x, N).T
y1, y2 = np.random.multivariate_normal(mean_y, cov_y, N).T

classA = np.column_stack((x1,x2)).T
classB = np.column_stack((y1,y2)).T 

X = np.append(classA[0], classB[0])
Y = np.append(classA[1], classB[1])
T =  np.append(np.ones(len(classA[0])), -np.ones(len(classB[0])))
bias = np.ones(2*N)

s = np.random.permutation(2*N)
patterns = np.concatenate(([X[s]], [Y[s]], [bias]), axis=0)
target =T[s]

print(patterns)
print(target)

# plt.plot(x1, x2,  'x')
# plt.plot(y1, y2, 'o')
# plt.axis('equal')
# plt.show()