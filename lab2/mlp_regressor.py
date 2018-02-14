import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

def data(noise):
    train = np.arange(0, 2*np.pi, 0.1)
    test = np.arange(0.05, 2*np.pi, 0.1)

    #Adding noise with variance=0.1
    if (noise==1):
        noise_train = np.random.normal(0, 0.1, len(train))
        noise_test = np.random.normal(0, 0.1, len(test))
        train = train + noise_train
        test = test + noise_test

    target_1 = sin_function(train)
    test_target_1 = sin_function(test)
    target_2 = square_function(train)
    test_target_2 = square_function(test)

    # # Shuffle data
    # index_train = np.random.permutation(len(train))
    # index_test = np.random.permutation(len(test))
    # #np.random.shuffle(train)
    # train = train[index_train]
    # test=test[index_test]
    # target_1 = target_1[index_train]
    # test_target_1=test_target_1[index_test]
    # target_2 = target_2[index_train]
    # test_target_2=test_target_2[index_test]

    return train, test, target_1, target_2, test_target_1, test_target_2

def sin_function(x):
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = np.sin(2*x[i])
    return y

def square_function(x):
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = np.sign(np.sin(2*x[i]))
    return y

def mlp_backprop(train, target, test, nodes, eta):
    nn = MLPRegressor(
        hidden_layer_sizes=(nodes,),  activation='logistic', solver='adam', alpha=0.001, batch_size='auto',
        learning_rate='constant', learning_rate_init= eta, power_t=0.5, max_iter=2000, shuffle=True,
        random_state=9, tol=0.0001, momentum=0.9, nesterovs_momentum=True,
        early_stopping=True, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    n = nn.fit(train, target)
    test_y = n.predict(test)
    return test_y

def main():
    eta = 0.0001
    sigma_value=0.2
    nodes= 60
    noise=0 #noise=0 without noise, noise=1 for a gaussian noise
    train, test, target_1, target_2, test_target_1, test_target_2 = data(noise)
        # eta = [0.0001, 0.001, 0.01, 0.1]    #Eta doesn't have any effect at all.
    type='sin'
    nodes = [15, 35, 45, 55, 63]
    mlp_errors = []
    errors_tot= []
    if type == 'sin':
        target = target_1
    if type == 'square':
        target = target_2
    train = train.reshape(-1,1) #columnvector
    test = test.reshape(-1,1)
    np.transpose(target)

    # for i in range(len(eta)):
    for j in range(len(nodes)):
        y_test =  mlp_backprop(train, target, test, nodes[j], eta = 0.01)
        # print(target, "target!")
        # print(y_test, "ytest")
        mlp_error = 0
        for k in range(len(target)):
            mlp_error += np.absolute((target[k]-y_test[k]))
        
        mlp_error = mlp_error/len(target)
        print(mlp_error)
    #         mlp_errors.append(mlp_error)
    #     errors_tot.append(mlp_errors)
    # for i in range(len(eta)):
    #     print('eta = ', eta[i])
    #     print(errors_tot[i])

if __name__ == "__main__":
    main()