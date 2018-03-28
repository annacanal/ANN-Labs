from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers
import data_handling


# this is the size of our encoded representations
encoding_dim = 50  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
#we initialize the weights to be small, random, normally distributed
#and the biases to zero
encoded = Dense(encoding_dim, activation='relu', kernel_initializer='random_normal',bias_initializer='zeros')(input_img)


# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid', kernel_initializer='random_normal',bias_initializer='zeros')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

#Now let's train our autoencoder to reconstruct MNIST digits.
#for training we use stochastic gradient descent as requested
sgd = optimizers.SGD(lr=0.5, momentum=0, decay=0, nesterov=False)
autoencoder.compile(optimizer=sgd, loss='binary_crossentropy')

from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()


#We will normalize all values between 0 and 1 and we will flatten the 28x28 images into vectors of size 784.
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

train,train_targets = data_handling.read_train_dataset()
test, test_targets = data_handling.read_test_dataset()

print(train[0])
print(train_targets[0])


#Now let's train our autoencoder for 50 epochs:
autoencoder.fit(train, train,
                epochs=100,
                batch_size=32, #default
                shuffle=True,
                validation_data=(test, test))

# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(test)
decoded_imgs = decoder.predict(encoded_imgs)

# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()