from sklearn.neural_network import MLPClassifier

# alpha = L2 penalty (regularization term) parameter.
# batch_size = when set to “auto”, batch_size=min(200, n_samples)
#learning_ rate = ‘constant’ is a constant learning rate given by ‘learning_rate_init
# suffle = Whether to shuffle samples in each iteration.

clf = MLPClassifier(hidden_layer_sizes=(1, ), activation=’relu’, solver=’adam’, alpha=0.0001, 
batch_size=’auto’, learning_rate=’constant’, learning_rate_init=0.001, max_iter=200, 
shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, 
nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)







def main():

if __name__ == "__main__":
    main()