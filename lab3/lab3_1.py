import numpy as np



def read_pictData():
    number_patterns = 11 #9 or 11 patterns of lenght = 1024
    patterns_matrix = np.zeros([number_patterns,1024])

    with open("pict (1).dat", "r") as f:
        # Read the whole file at once
        patterns_line = f.read()
    patterns_line = patterns_line.split(",")
    for i in range(number_patterns):
        for j in range(1024):
            position = j+1024*i
            patterns_matrix[i][j] = patterns_line[position]

    return patterns_matrix

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
