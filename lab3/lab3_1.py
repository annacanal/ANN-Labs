import numpy as np


#make test patterns
patterns=np.array([[-1, -1, 1, -1, 1, -1, -1, 1],[-1, -1, -1, -1, -1, 1, -1, -1],[-1, 1, 1, -1, -1, 1, -1, 1]])
Nodes = 8



#way 1
W1 = np.zeros((Nodes,Nodes))
for i in range(W1.shape[0]):
    for j in range(W1.shape[1]):
        for k in range(patterns.shape[0]):
            W1[i][j] += (1/Nodes) * patterns[k][i] * patterns[k][j]

#way 2
W2 = np.zeros((Nodes,Nodes))
for k in range(patterns.shape[0]):
    W2 += (1/Nodes) * ( np.outer(np.transpose(patterns[k]), patterns[k])  )


print(W1)
print(W2)
print(W1-W2)