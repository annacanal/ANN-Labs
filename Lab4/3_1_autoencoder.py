from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers
import data_handling
import numpy as np
import matplotlib.pyplot as plt


def get_mae_error_per_epoch(train, test, epochs):
    mae_error = np.zeros(epochs-1)

    for ep in range(1, 4, 1):
        encoder, decoder, autoencoder = create_autoencoder(nodes)

        # Now let's train our autoencoder for 50 epochs:
        history = autoencoder.fit(train, train,
                        epochs=ep,
                        batch_size=1,  # default
                        verbose=2,
                        shuffle=True,
                        validation_split=0.3,
                        # validation_data=(test, test)
                        )
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
    sgd = optimizers.SGD(lr=0.1, momentum=0, decay=0, nesterov=False)
    autoencoder.compile(optimizer=sgd, loss='mean_squared_error', metrics=['mae'])

    return (encoder, decoder, autoencoder)

epochs = 21
# mae_errors = np.zeros((4, epochs-1))

# this is the size of our encoded representations
for i,nodes in enumerate([50,75,100,150]):


    #We use the prenormalized data given to us
    train,train_targets = data_handling.read_train_dataset()
    test, test_targets = data_handling.read_test_dataset()

    mae_errors = np.zeros(epochs - 1)
    mae_errors = get_mae_error_per_epoch(train, test, epochs)

    x = np.arange(1, epochs, 1)
    plt.plot(x, mae_errors, label=('Nodes = '+str(nodes)))
plt.legend()
plt.ylabel('Mean Error')
plt.xlabel('Epochs')
plt.savefig('speedup.png')
plt.show()

# # encode and decode some digits
# # note that we take them from the *test* set
# encoded_imgs = encoder.predict(test)
# decoded_imgs = decoder.predict(encoded_imgs)
#
#
# n = 10  # how many digits we will display
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # display original
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(test[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
#     # display reconstruction
#     ax = plt.subplot(2, n, i + 1 + n)
#     plt.imshow(decoded_imgs[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()