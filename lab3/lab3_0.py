import numpy as np
import matplotlib.pyplot as plt


def patterns():
    x1 = np.array([0, 0, 1, 0, 1, 0, 0, 1])
    x2 = np.array([0, 0, 0, 0, 0, 1, 0, 0])
    x3 = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    # pattern = np.concatenate(x1,x2)
    return x1, x2, x3

def binary_bipolar(x):
    for i in range(len(x)):
        if x[i]==0:
            x[i] = -1
        else:
            x[i] = 1
    return x

def bipolar_binary(x):
    for i in range(len(x)):
        if x[i]==-1:
            x[i] = 0
        else:
            x[i] = 1
    return x

def weight_matrix(pattern):
    matrix = np.zeros(())
    return matrix


def main():
    x1, x2, x3 = patterns()
    # x1_bin = binary_bipolar(x1)
    # print(x1_bin)
    # x1_bip = bipolar_binary(x1_bin)
    # print(x1_bip)

if __name__ == "__main__":
    main()