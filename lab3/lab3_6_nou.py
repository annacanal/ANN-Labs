import numpy as np
import matplotlib.pyplot as plt

def weight_matrix(nodes, patterns, rho):
    W = np.zeros((nodes, nodes))
    for k in range(len(patterns)):
        W += (1 / nodes) * (np.outer(np.transpose(patterns[k]-rho), patterns[k]-rho))
        #W += (np.outer(np.transpose(patterns[k] - rho), patterns[k] - rho))
    return W

def update(W, input_pattern,teta,idx):
    output = input_pattern
    output[idx] = np.sum(W[idx] * input_pattern.T)
    output[output-teta >= 0] = 1
    output[output-teta < 0] = 0
    new_output = 0.5 + 0.5*output

    return new_output


def seq_update_book(W, input_pattern, nodes, teta):
    output = np.zeros(nodes)
    order = np.arange(nodes)
    np.random.shuffle(order)
    for i in order:
        diff = np.sum(W[i,:]*input_pattern) - teta
        if diff>=0:
            output[i] = 1
        else:
            output[i] = -1
    new_output = 0.5 + 0.5 * output

    return new_output


def make_bias(original_pattern, percentage):
    #as half of the randomly selected elements will already be ones, we need to select double of them
    percentage = percentage * 2
    biased_pattern = np.copy(original_pattern)
    # how many bits do we need to flip
    elements_num = biased_pattern.size * percentage / 100
    elements_num = int(elements_num)
    # select that many random indexes
    indexes = np.random.choice(original_pattern.size, elements_num, replace=False)
    # flip the bit
    for i, idx in enumerate(indexes):
        if biased_pattern[idx] == -1:
            biased_pattern[idx] = 1

def main():
    N = 200 #Number of cells in every pattern
    P = 100 # Number of patterns
    act_perc = 0.1 #Active proportion of cells (rho)
    all_patterns = np.zeros([P, N]) #Create P patterns with only a 10% of activation
    # Generate patterns with a 10% of activation
    for i in range (P):
        index =np.random.permutation(N)
        num = np.round(N*act_perc)
        index = index[0:int(num)]
        all_patterns[i][index] = 1

    Max_bias_val = 15  #Max bias value that will be use
    bias_value= np.arange(0.1,1, 0.1)
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
            patterns = all_patterns[0:p+1]
           # print(patterns)
            #train
            W = weight_matrix(N, patterns, rho)
            saved = 0  # Number of saved patterns
            original_pat = patterns#.T
            #for i in range(N):
            for j in range(p + 1):
                output = seq_update_book(W, original_pat, N,bias)#,i)
                diff = np.sum(abs(original_pat - output))
                if diff == 0:
                    saved = saved + 1
            capacity_percentage.append(saved*100/(p+1))
        print(capacity_percentage)
        plt.title("Patterns with "+str(act_perc*100)+"% activity, capacity/patterns with bias= " + str(bias))
        plt.plot(np.arange(p+1),capacity_percentage)
        plt.xlabel('Number of patterns')
        plt.ylabel('Capacity percentage')
        plt.show()

if __name__ == "__main__":
    main()