from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers
import data_handling
import matplotlib.pyplot as plt


# encoding_dim = 50

input_img = Input(shape=(784,))

#------ 2 hidden layers----------------
encoded = Dense(128, activation = 'relu', kernel_initializer='random_normal',bias_initializer='zeros')(input_img)
encoded = Dense(32, activation = 'relu')(encoded)

decoded = Dense(128, activation = 'relu')(encoded)
decoded = Dense(784, activation='sigmoid', kernel_initializer='random_normal',bias_initializer='zeros')(decoded)

# #------ 3 hidden layers -----------------
# encoded = Dense(128, activation = 'relu', kernel_initializer='random_normal',bias_initializer='zeros')(input_img)
# encoded = Dense(64, activation = 'relu')(encoded)
# encoded = Dense(32, activation = 'relu')(encoded)

# decoded = Dense(32, activation = 'relu')(encoded)
# decoded = Dense(128, activation = 'relu')(decoded)
# decoded = Dense(784, activation='sigmoid', kernel_initializer='random_normal',bias_initializer='zeros')(decoded)
# #----------------------------------------

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)
encoded_input = Input(shape=(128,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))


sgd = optimizers.SGD(lr=0.2, momentum=0, decay=0, nesterov=False)
autoencoder.compile(optimizer=sgd, loss='mean_squared_error', metrics=['mae'])

train, train_targets = data_handling.read_train_dataset()
test, test_targets = data_handling.read_test_dataset()

autoencoder.fit(train, train,
                epochs=50,
                batch_size=300, #default
                shuffle=True,
                validation_data=(test, test))

encoded_imgs = encoder.predict(test)
decoded_imgs = decoder.predict(encoded_imgs)


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