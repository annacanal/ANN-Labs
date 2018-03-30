from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers
import data_handling
import numpy as np
import matplotlib.pyplot as plt


def get_mae_error_per_epoch(train, test, nodes, epochs):
    mae_error = np.zeros(epochs-1)

    for ep in range(1, epochs, 1):
        encoder, decoder, autoencoder = create_and_fit_autoencoder(ep, nodes, train)
        what = autoencoder.evaluate(x=test, y=test, verbose=2)
        mae_error[ep-1] = what[1]

    return mae_error

def create_autoencoder(encoding_nodes):
    # this is our input placeholder
    input_img = Input(shape=(784,))
    # "encoded" is the encoded representation of the input
    # we initialize the weights to be small, random, normally distributed
    # and the biases to zero
    encoded = Dense(encoding_nodes, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros')(
        input_img)

    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(784, activation='sigmoid', kernel_initializer='random_normal', bias_initializer='zeros')(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)

    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)

    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_nodes,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    # Now let's train our autoencoder to reconstruct MNIST digits.
    # for training we use stochastic gradient descent as requested
    sgd = optimizers.SGD(lr=0.3, momentum=0, decay=0, nesterov=False)
    autoencoder.compile(optimizer=sgd, loss='mean_squared_error', metrics=['mae'])

    return (encoder, decoder, autoencoder)

def plot_error_per_epoch(epochs, diff_nodes, train, test):

    # mae_errors = np.zeros((4, epochs-1))

    # this is the size of our encoded representations
    for i,nodes in enumerate(diff_nodes):

        mae_errors = np.zeros(epochs - 1)
        mae_errors = get_mae_error_per_epoch(train, test, nodes, epochs)

        x = np.arange(1, epochs, 1)
        plt.plot(x, mae_errors, label=('Nodes = '+str(nodes)))
    plt.legend()
    plt.ylabel('Mean Error')
    plt.xlabel('Epochs')
    plt.savefig('error_per_epoch.png')
    plt.show()

def create_and_fit_autoencoder(epochs, nodes, train, test):
    encoder, decoder, autoencoder = create_autoencoder(nodes)

    # Now let's train our autoencoder for ep epochs:
    history = autoencoder.fit(train, train,
                              epochs=epochs,
                              batch_size=1,  # default
                              verbose=2,
                              shuffle=True,
                              #validation_split=0.3,
                              validation_data=(test, test)
                              )

    return (encoder, decoder, autoencoder, history)

def evaluate_autoencoder(epochs, nodes, train, test):
    encoder, decoder, autoencoder = create_and_fit_autoencoder(epochs, nodes, train)
    what = autoencoder.evaluate(x=test, y=test, verbose=2)
    mae_error = what[1]

    return mae_error

def evaluate_autoencoders(epochs, diff_nodes, train, test):

    for i,nodes in enumerate(diff_nodes):
        error = evaluate_autoencoder(epochs, nodes, train, test)
        print('error for nodes ', nodes, ': ', error)


def plot_digits(predictions, test):
    example_digits_indexs = [18, 3, 7, 0, 2, 1, 14, 8, 6, 5]  # indexs in the test partition digits: 0,1,2,3,4,5,6,7,8,9

    plt.figure(figsize=(20, 20))

    for index, i in enumerate(example_digits_indexs):
        plt.subplot(10, 5, 5 * index + 1)
        plt.imshow(test[i].reshape((28, 28)), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.subplot(10, 5, 5 * index + 2)
        plt.imshow(predictions[0][i].reshape((28, 28)), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.subplot(10, 5, 5 * index + 3)
        plt.imshow(predictions[1][i].reshape((28, 28)), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.subplot(10, 5, 5 * index + 4)
        plt.imshow(predictions[2][i].reshape((28, 28)), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.subplot(10, 5, 5 * index + 5)
        plt.imshow(predictions[3][i].reshape((28, 28)), cmap=plt.cm.gray_r,
                   interpolation='nearest')
    plt.savefig('img_of_digits.png')
    plt.show()

def plot_digits_for_diff_nodes(diff_nodes, epochs, train, test):
    predictions = []
    for nodes in diff_nodes:
        encoder, decoder, autoencoder = create_and_fit_autoencoder(epochs, nodes, train)
        error = autoencoder.evaluate(x=test, y=test, verbose=2)
        print('error for nodes ', nodes, ': ', error[1])

        encoded_imgs = encoder.predict(test)
        decoded_imgs = decoder.predict(encoded_imgs)

        predictions.append(decoded_imgs)

    plot_digits(predictions, test)

def plot_everything(diff_nodes, epochs, train, test):
    histories = []
    predictions = []
    for nodes in diff_nodes:
        encoder, decoder, autoencoder, history = create_and_fit_autoencoder(epochs, nodes, train, test)
        print("MAE ERROR for ", nodes, "nodes: ", (autoencoder.evaluate(x=test, y=test, verbose=2))[1])

        histories.append(history)

        encoded_imgs = encoder.predict(test)
        decoded_imgs = decoder.predict(encoded_imgs)
        predictions.append(decoded_imgs)

        the_weights = (autoencoder.get_weights())[2]


        if nodes == 50 or nodes == 100:
            plt.figure(figsize=(10, 10))
            for i in range(nodes):
                plt.subplot(10, 10, i + 1)
                plt.imshow(the_weights[i].reshape((28, 28)), cmap=plt.cm.gray_r,
                           interpolation='nearest')
                plt.xticks(())
                plt.yticks(())
            plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
            plt.savefig(str(nodes)+'_weights_autoencoder.png')
            # plt.show()
            plt.close()

    for i,nodes in enumerate(diff_nodes):

        mae_errors = histories[i].history['val_mean_absolute_error']
        x = np.arange(1, epochs+1, 1)
        plt.plot(x, mae_errors, label=('Nodes = '+str(nodes)))
    plt.legend()
    plt.ylabel('Mean Error')
    plt.xlabel('Epochs')
    plt.savefig('error_per_epoch.png')
    plt.show()
    plt.close()

    plot_digits(predictions, test)


def main():
    # We use the prenormalized data given to us
    train, train_targets = data_handling.read_train_dataset()
    test, test_targets = data_handling.read_test_dataset()

    epochs = 75
    diff_nodes = [50, 75, 100, 150]

    plot_everything(diff_nodes, epochs, train, test)

main()
