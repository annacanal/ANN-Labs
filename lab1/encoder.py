import numpy as np
import Two_Layer_Perceptron
import Evaluation


def possible_inputs(num_of_combinations):
    inputs = -1 * np.ones((num_of_combinations,num_of_combinations))
    for x in range(0,num_of_combinations):
        inputs[x,x] = 1

    return inputs



def encoder_miscl_ratio(outputs, targets):
    miscl = 0
    outputs_r = np.round(outputs)
    for i, x in enumerate(targets.T):
        if not(all(x == (outputs_r.T)[i])):
            miscl += 1
    ratio = miscl / np.shape(targets)[1]
    return ratio

def backforward_prop(patterns, targets, n_nodes):
    epochs = 6000
    eta = 0.1
    alpha= 0.9
    deltaW = 0
    deltaV = 0

    X = patterns
    W = Two_Layer_Perceptron.W_init(n_nodes[0], np.size(X, 0))
    V = Two_Layer_Perceptron.W_init(n_nodes[1], n_nodes[0]+1)

    errors_miscl=[]
    errors_mse=[]
    for i in range(epochs):
        H, O = Two_Layer_Perceptron.forward_pass(X, W, V)
        deltaO, deltaH = Two_Layer_Perceptron.backward_pass(X, W, V, H, O, targets, n_nodes)
        deltaW, deltaV = Two_Layer_Perceptron.weight_update(eta, deltaO, deltaH, X, H, alpha, deltaW, deltaV)
        W = W + deltaW
        V = V + deltaV


        error_miscl = encoder_miscl_ratio(O, targets)
        error_mse = Evaluation.mean_sq_error(O, targets)
        errors_miscl.append(error_miscl)
        errors_mse.append(error_mse)
    iterations = np.arange(epochs)
    print(W, 'this is W')
    print(V, 'this is V')

    # print(np.round(O))
    # print()
    Evaluation.Plot_error_curve("MSE/iteration in learning", iterations, errors_mse)
    Evaluation.Plot_error_curve("Missclassification/iteration in learning", iterations, errors_miscl)




#possible_inputs()

num_of_combinations = 8
n_nodes = [3,8]


patterns = possible_inputs(num_of_combinations)

# for i in range(100):
#     patterns = np.hstack((patterns, possible_inputs(num_of_combinations)))
#
# np.transpose(np.random.shuffle(np.transpose(patterns)))


targets = patterns

bias = np.ones(np.shape(patterns)[1])
patterns = np.concatenate((patterns, [bias]), axis = 0)

backforward_prop(patterns, targets, n_nodes)