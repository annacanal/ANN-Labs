import numpy as np

def possible_inputs(num_of_combinations):
    inputs = -1 * np.ones((num_of_combinations,num_of_combinations))
    for x in range(0,num_of_combinations):
        inputs[x,x] = 1

    return inputs

def W_init(Wshape):
    W = np.zeros(Wshape)
    W = np.random.normal(0,0.001,W.shape)
   # print(W)
    return W

#possible_inputs()

num_of_combinations = 8

allinputs = possible_inputs(num_of_combinations)
W1 = W_init((num_of_combinations, 3))
W2 = W_init((3, num_of_combinations))


