# #this file can used for decompression separately, this also needs to train the autoencoder
# #as we train already while compression so for reducing time , i have concated this portion into compression part
#
# import os
# import matplotlib.pyplot as plt
# import numpy as np
# import keras
# from PIL import Image
# from keras.datasets import mnist
# from keras.models import Model, Sequential
# from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
# from keras import regularizers
# from keras.preprocessing import image
# from keras.src.saving.saving_api import load_model
#
# # Load the encoded images
# encoded_imgs = []
# for filename in os.listdir("./"):
#     if filename.startswith("encoded_img"):
#         encoded_img = np.load("./" + filename)
#         encoded_imgs.append(encoded_img)
# encoded_imgs = np.array(encoded_imgs)
#
# x_train_path = "./"
#
# x_train = []
# for filename in os.listdir(x_train_path):
#     if filename.startswith("shuffled_img"):
#         # img = image.load_img(x_train_path + filename, target_size=(256, 256))
#         img = np.load(x_train_path + filename)
#         x_train.append(image.img_to_array(img))
# x_train = np.array(x_train)
#
# x_test = []
# for filename in os.listdir(x_train_path):
#     if filename.startswith("shuffled_img"):
#         # img = image.load_img(x_train_path + filename, target_size=(256, 256))
#         img = np.load(x_train_path + filename)
#         x_test.append(image.img_to_array(img))
# x_test = np.array(x_test)
#
# x_train = x_train.astype('float32') / 255.0
# x_test = x_test.astype('float32') / 255.0
#
# x_train = x_train.reshape(len(x_train), np.prod(x_train.shape[1:]))
# x_test = x_test.reshape(len(x_test), np.prod(x_test.shape[1:]))
#
# input_dim = x_train.shape[1]
#
# encoding_dim = 100
#
# autoencoder = Sequential()
#
# # Encoder Layers
# autoencoder.add(Dense(4 * encoding_dim, input_shape=(input_dim,), activation='relu'))
# autoencoder.add(Dense(2 * encoding_dim, activation='relu'))
# autoencoder.add(Dense(encoding_dim, activation='relu'))
#
# # Decoder Layers
# autoencoder.add(Dense(2 * encoding_dim, activation='relu'))
# autoencoder.add(Dense(4 * encoding_dim, activation='relu'))
# autoencoder.add(Dense(input_dim, activation='sigmoid'))
#
# #autoencoder.summary()
#
# input_img = Input(shape=(input_dim,))
# encoder_layer1 = autoencoder.layers[0]
# encoder_layer2 = autoencoder.layers[1]
# encoder_layer3 = autoencoder.layers[2]
# encoder = Model(input_img, encoder_layer3(encoder_layer2(encoder_layer1(input_img))))
#
# # encoder.summary()
#
# # Define the decoder model
# encoded_input = Input(shape=(encoding_dim,))
# decoder_layer1 = autoencoder.layers[-3]
# decoder_layer2 = autoencoder.layers[-2]
# decoder_layer3 = autoencoder.layers[-1]
# decoder = Model(encoded_input, decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))
#
# # Display the summary of the decoder model
# decoder.summary()
#
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# autoencoder.fit(x_train, x_train,
#                 epochs=50,
#                 batch_size=32,
#                 validation_data=(x_test, x_test))
#
# decoded_imgs = decoder.predict(encoded_imgs)
#
# for i, decoded_img in enumerate(decoded_imgs):
#     filename = f"decoded_img_{i}.npy"
#     np.save(filename, decoded_img)
#
# num_images = 1
# np.random.seed(42)
# random_test_indices = np.random.choice(x_test.shape[0], size=num_images, replace=False)
#
# plt.figure(figsize=(18, 2*num_images))
#
# for i, image_idx in enumerate(random_test_indices):
#     ax = plt.subplot(3, num_images, num_images + i + 1)
#     plt.imshow(encoded_imgs[image_idx].reshape(10, 10))
#     plt.gray()
#     ax.get_xaxis().set_visible(True)
#     ax.get_yaxis().set_visible(True)
#     ax.set_title("Decrypted(Compresed)")
#
#     ax = plt.subplot(3, num_images, 2 * num_images + i + 1)
#     plt.imshow(decoded_imgs[image_idx].reshape(256, 256, 3))
#     plt.gray()
#     ax.get_xaxis().set_visible(True)
#     ax.get_yaxis().set_visible(True)
#     ax.set_title("Decompressed")
#
# plt.tight_layout()
# plt.show()
