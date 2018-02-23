import numpy as np
import matplotlib.pyplot as plt

def weight_matrix(nodes, patterns, rho):
    W = np.zeros((nodes, nodes))
    for k in range(len(patterns)):
        W += (1 / nodes) * (np.outer(np.transpose(patterns[k]-rho), patterns[k]-rho))
    return W

def update(W, input_pattern,teta):
    output = np.sum(W * input_pattern, axis=1)
    output[output-teta >= 0] = 1
    output[output-teta < 0] = -1
    new_output = 0.5 + 0.5*output

    return new_output


def main():
    N = 200 #Number of cells in every pattern
    P = 100 # Number of patterns
    act_perc = 0.1 #Active proportion of cells
    all_patterns = np.zeros([P, N]) #Create P patterns with only a 10% of activation
    # Generate patterns with a 10% of activation
    for i in range (P):
        # index = randperm(N)
        # index = index(1:round(N*act_perc))
        idx = np.random.randint(1, N*act_perc)
        all_patterns[i][idx] = 1


    Max_bias_val = 15  #Max bias value that will be use
   # step_val = 3 #Step between bias values during the experimentation
    for mu in range(all_patterns.shape[0]):
        for i in range(all_patterns.shape[1]):
             rho = all_patterns[mu][i]*(1 / (N*P))  # Average activity: average value of the cells in all patterns

    #train patterns
    W = weight_matrix(N, all_patterns,rho)
    for bias in range(Max_bias_val):
        output = update(W, all_patterns[1], bias)


if __name__ == "__main__":
    main()
