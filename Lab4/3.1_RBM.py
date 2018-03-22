from sklearn.neural_network import BernoulliRBM
import numpy as np
from sklearn import linear_model, datasets, metrics
import data_handling
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def train_classifier(train,train_targets, learning_rate,n_iter,n_components):
    clf = BernoulliRBM(n_components=n_feat)

    RBM_feature_data = clf.fit(train,train_targets)
    BernoulliRBM(batch_size=10, learning_rate=learning_rate, n_components=n_components, n_iter=n_iter,
                 random_state=None, verbose=0)
    return RBM_feature_data


def main():
    #Read data
    train,train_targets = data_handling.read_train_dataset()
    test, test_targets = data_handling.read_test_dataset()

    # Hyper-parameters. These were set by cross-validation,
    # using a GridSearchCV. Here we are not performing cross-validation to
    # save time.
    learning_rate = 0.06
    n_iter = [10, 20]
    # More components tend to give better prediction performance, but larger
    # fitting time
    n_components = 100

    #Training
    RBM_feature_data = train_classifier(train,train_targets, learning_rate,n_iter,n_components)

    # Evaluation
    print()
    print(RBM_feature_data)


if __name__ == "__main__":
    main()