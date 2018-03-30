from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers
import data_handling
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from keras.utils import np_utils
from sklearn.metrics import precision_recall_fscore_support

def evaluate(test_labels, predictions):
    precision, recall, f1score, support = precision_recall_fscore_support(test_labels, predictions)
    precision1=np.mean(precision)
    recall1=np.mean(recall)
    f1score1=np.mean(f1score)
    print("eval")
    # print('precision: {}'.format(precision))
    # print('recall: {}'.format(recall))
    # print('fscore: {}'.format(f1score))
    print('mean precision: {}'.format(precision1))
    print('mean recall: {}'.format(recall1))
    print('mean fscore: {}'.format(f1score1))



#-----------pre-training------------------------

input_img = Input(shape=(784,))

#3 hidden layers: 
encoded = Dense(150, activation = 'relu', kernel_initializer='random_normal', bias_initializer='zeros')(input_img)
encoded = Dense(120, activation = 'relu', kernel_initializer='random_normal', bias_initializer='zeros')(encoded)
encoded = Dense(90, activation = 'relu', kernel_initializer='random_normal', bias_initializer='zeros')(encoded)

decoded = Dense(120, activation = 'relu', kernel_initializer='random_normal', bias_initializer='zeros')(encoded)
decoded = Dense(150, activation = 'relu', kernel_initializer='random_normal', bias_initializer='zeros')(decoded)
decoded = Dense(784, activation='sigmoid', kernel_initializer='random_normal', bias_initializer='zeros')(decoded)

#Autoencoder: 
autoencoder = Model(input_img, decoded)

#Encoder:
encoder = Model(input=input_img, output=encoded)

#Decoder
encoded_input_1 = Input(shape=(90,))
encoded_input_2 = Input(shape=(120,))
encoded_input_3 = Input(shape=(150,))

decoder_layer_1 = autoencoder.layers[-3]
decoder_layer_2 = autoencoder.layers[-2]
decoder_layer_3 = autoencoder.layers[-1]

decoder_1 = Model(input = encoded_input_1, output = decoder_layer_1(encoded_input_1))
decoder_2 = Model(input = encoded_input_2, output = decoder_layer_2(encoded_input_2))
decoder_3 = Model(input = encoded_input_3, output = decoder_layer_3(encoded_input_3))

#Training in pre-training phase
sgd = optimizers.SGD(lr=0.3, momentum=0, decay=0, nesterov=False)
autoencoder.compile(optimizer=sgd, loss='mean_squared_error', metrics=['mae'])

train, train_targets = data_handling.read_train_dataset()
test, test_targets = data_handling.read_test_dataset()
train_targets = np_utils.to_categorical(train_targets)
test_targets = np_utils.to_categorical(test_targets)

autoencoder.fit(train, train,
                epochs=75,
                batch_size=1, #default
                shuffle=True,
                validation_data=(test, test))

#-------------pretraining done, adding classifier-------------------

#adding one more "layer"
decoded = Dense(10, activation='sigmoid', kernel_initializer='random_normal', bias_initializer='zeros')(decoded)

classifier = Model(input = input_img, output = decoded)

#training classifier
sgd = optimizers.SGD(lr=0.3, momentum=0, decay=0, nesterov=False)
classifier.compile(optimizer=sgd, loss='mean_squared_error', metrics=['mae'])

hist = classifier.fit(train, train_targets, 
                epochs = 75, 
                batch_size = 1, 
                shuffle = True,
                validation_data=(test, test_targets))

predictions = classifier.predict(test)

encoded_imgs = encoder.predict(test)
decoded_imgs = decoder_1.predict(encoded_imgs)
decoded_imgs = decoder_2.predict(decoded_imgs)
decoded_imgs = decoder_3.predict(decoded_imgs)

evaluate(test, decoded_imgs.round()) #classifier or decdoded_imgs?

#--------------------------------------------------------------------------------

# n = 10  
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(test[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

#     ax = plt.subplot(2, n, i + 1 + n)
#     # plt.imshow(decoded_imgs[i].reshape(28, 28))
#     plt.imshow(decoded_imgs[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

# # np.save('3_hidden_layers_150_120_90_epoch50_lr01_batchsize1.npy', hist.history)
# np.save('test_3_layers_75epochs_0.3lr.npy', hist.history)

# # plt.figure(2)
# # plt.plot(hist.history['mean_absolute_error'])

# plt.show()


# Plotting
weights_3 = decoder_3.get_weights()[0]
weights_2 = decoder_2.get_weights()[0]
weights_1 = decoder_1.get_weights()[0]

print(np.array(weights_3).shape)
print(np.array(weights_2).shape)
print(np.array(weights_1).shape)

# print(np.array(weights[1]).shape)

plt.figure(figsize=(10, 10))
for i in range(150):
    plt.subplot(10, 15, i + 1)
    plt.imshow(weights_3[i].reshape((28, 28)), cmap=plt.cm.gray_r,
                interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('150 components extracted by autoencoder layer 1', fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
plt.show()

plt.figure(figsize=(10, 10))
for i in range(120):
    plt.subplot(10, 15, i + 1)
    plt.imshow(weights_2[i].reshape((10, 15)), cmap=plt.cm.gray_r,
                interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('120 components extracted by autoencoder layer 2', fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
plt.show()

plt.figure(figsize=(10, 10))
for i in range(90):
    plt.subplot(10, 15, i + 1)
    plt.imshow(weights_1[i].reshape((10, 12)), cmap=plt.cm.gray_r,
                interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('90 components extracted by autoencoder layer 3', fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
plt.show()