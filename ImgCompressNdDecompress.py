import os
import  subprocess
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.preprocessing import image

x_train_path = "./"

x_train = []
for filename in os.listdir(x_train_path):
    if filename.startswith("shuffled_img"):
        img = np.load(x_train_path + filename)
        x_train.append(image.img_to_array(img))
x_train = np.array(x_train)

x_test = []
for filename in os.listdir(x_train_path):
    if filename.startswith("shuffled_img"):
        img = np.load(x_train_path + filename)
        x_test.append(image.img_to_array(img))
x_test = np.array(x_test)


x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

print(x_train.shape, x_test.shape)

print(x_train.shape, x_test.shape)
x_train = x_train.reshape(len(x_train), np.prod(x_train.shape[1:]))
x_test = x_test.reshape(len(x_test), np.prod(x_test.shape[1:]))

print(x_train.shape, x_test.shape)

input_dim = x_train.shape[1]

encoding_dim = 100

autoencoder = Sequential()

# Encoder Layers
autoencoder.add(Dense(4 * encoding_dim, input_shape=(input_dim,), activation='relu'))
autoencoder.add(Dense(2 * encoding_dim, activation='relu'))
autoencoder.add(Dense(encoding_dim, activation='relu'))

# Decoder Layers
autoencoder.add(Dense(2 * encoding_dim, activation='relu'))
autoencoder.add(Dense(4 * encoding_dim, activation='relu'))
autoencoder.add(Dense(input_dim, activation='sigmoid'))

# autoencoder.add(Dense(input_dim, activation='sigmoid'))

autoencoder.summary()

input_img = Input(shape=(input_dim,))
encoder_layer1 = autoencoder.layers[0]
encoder_layer2 = autoencoder.layers[1]
encoder_layer3 = autoencoder.layers[2]
encoder = Model(input_img, encoder_layer3(encoder_layer2(encoder_layer1(input_img))))

encoder.summary()

# Define the decoder model
encoded_input = Input(shape=(encoding_dim,))
decoder_layer1 = autoencoder.layers[-3]
decoder_layer2 = autoencoder.layers[-2]
decoder_layer3 = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))

# Display the summary of the decoder model

decoder.summary()

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(x_train, x_train,
                epochs=60,
                batch_size=32,
                validation_data=(x_test, x_test))


encoded_imgs = encoder.predict(x_test)


for i, encoded_img in enumerate(encoded_imgs):
    filename = f"encoded_img_{i}.npy"
    np.save(filename, encoded_img)


num_images = 3
np.random.seed(42)
random_test_indices = np.random.choice(x_test.shape[0], size=num_images, replace=False)

plt.figure(figsize=(18, 2*num_images))

for i, image_idx in enumerate(random_test_indices):
    ax = plt.subplot(3, num_images, i + 1)
    plt.imshow(x_test[image_idx].reshape(256, 256, 3))
    plt.gray()
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.set_title("Shuffled Image")

    ax = plt.subplot(3, num_images, num_images + i + 1)
    plt.imshow(encoded_imgs[image_idx].reshape(10, 10))
    plt.gray()
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.set_title("Compressed")

plt.tight_layout()
plt.show()


subprocess.run(["python", "ImgEncryption.py"])
subprocess.run(["python", "ImgDecryption.py"])



# Load the encoded  images after decryption

encoded_imgs_after_decryption = []
for filename in os.listdir("./"):
    if filename.startswith("decrypted_img"):
        encoded_img_after_decryption = np.load("./" + filename)
        encoded_imgs_after_decryption.append(encoded_img_after_decryption)
encoded_imgs_after_decryption = np.array(encoded_imgs_after_decryption)


decoded_imgs = decoder.predict(encoded_imgs_after_decryption)


for i, decoded_img in enumerate(decoded_imgs):
    filename = f"decoded_img_{i}.npy"
    np.save(filename, decoded_img)

plt.figure(figsize=(18, 2*num_images))


for i, image_idx in enumerate(random_test_indices):
    ax = plt.subplot(3, num_images, i + 1)
    plt.imshow(encoded_imgs[image_idx].reshape(10, 10))
    plt.gray()
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.set_title("Decrypted(Compresed)")

    ax = plt.subplot(3, num_images, num_images + i + 1)
    plt.imshow(decoded_imgs[image_idx].reshape(256, 256, 3))
    plt.gray()
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.set_title("Decompressed")

plt.tight_layout()
plt.show()
