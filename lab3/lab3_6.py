import numpy as np
import matplotlib.pyplot as plt

def weight_matrix(nodes, patterns, rho):
    W = np.zeros((nodes, nodes))
    for k in range(len(patterns)):
        #W += (1 / nodes) * (np.outer(np.transpose(patterns[k]-rho), patterns[k]-rho))
        W += (np.outer(np.transpose(patterns[k] - rho), patterns[k] - rho))
    return W

def update(W, input_pattern,teta,idx):
    output = input_pattern
    output[idx] = np.sum(W[idx] * input_pattern.T)
    output[output-teta >= 0] = 1
    output[output-teta < 0] = 0
    new_output = 0.5 + 0.5*output

    return new_output

def main():
    N = 20 #Number of cells in every pattern
    P = 10 # Number of patterns
    act_perc = 0.05 #Active proportion of cells (rho)
    all_patterns = np.zeros([P, N]) #Create P patterns with only a 10% of activation
    # Generate patterns with a 10% of activation
    for i in range (P):
        index =np.random.permutation(N)
        num = np.round(N*act_perc)
        index = index[0:int(num)]
        all_patterns[i][index] = 1

    Max_bias_val = 15  #Max bias value that will be use
    bias_value= np.arange(0.1,0.5, 0.05)
   # step_val = 3 #Step between bias values during the experimentation
    rhos=[]
    for mu in range(all_patterns.shape[0]):
        for i in range(all_patterns.shape[1]):
             rhos.append(all_patterns[mu][i]*(1 / (N*P)))  # Average activity: average value of the cells in all patterns
    rho= np.sum(rhos,axis=0 )
    print(rho)
    for bias in bias_value:
        capacity_percentage = []
        for p in range(all_patterns.shape[0]):
            patterns = all_patterns[p]
           # print(patterns)
            #train
            W = weight_matrix(N, patterns, rho)
            saved = 0  # Number of saved patterns
            original_pat = patterns#.T
            for i in range(N):
                output = update(W, original_pat, bias,i)
           # output = sync_update(W, original_pat, bias)
            diff = np.sum(abs(original_pat - output))
            if diff == 0:
                saved = saved + 1
            capacity_percentage.append(saved*100/(p+1))
        print(capacity_percentage)
        plt.title("Patterns with 10% activity, capacity/patterns with bias= " + str(bias))
        plt.plot(np.arange(p+1),capacity_percentage)
        plt.xlabel('Number of patterns')
        plt.ylabel('Capacity percentage')
       # plt.show()

if __name__ == "__main__":
    main()

