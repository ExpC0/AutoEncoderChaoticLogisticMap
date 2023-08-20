import os
import numpy as np
from matplotlib import pyplot as plt

import chaoticLogisticMap as CLmap

# Load the encoded images
encoded_imgs = []

for filename in os.listdir("./"):
    if filename.startswith("encoded_img"):
        encoded_img = np.load("./" + filename)
        encoded_imgs.append(encoded_img)

encoded_imgs = np.array(encoded_imgs)

# Get the size of the encoded images
size = encoded_imgs[0].shape[0]

# Generate the chaotic sequence for the key
key = CLmap.keygen(0.971, 3.9159, size)

# Convert key to a NumPy array and then to np.uint8 data type
key = np.array(key, dtype=np.uint8)

encrypted_imgs = []

for encoded_img in encoded_imgs:
    # Convert encoded_img to np.uint8 data type
    encoded_img_uint8 = encoded_img.astype(np.uint8)

    # Perform bitwise XOR operation
    encrypted_img = np.bitwise_xor(encoded_img_uint8, key)
    encrypted_imgs.append(encrypted_img)

# Save the encrypted images
for i, encrypted_img in enumerate(encrypted_imgs):
    filename = f"encrypted_img_{i}.npy"
    np.save(filename, encrypted_img)

print("Image is encrypted & saved!")

num_images  = 3
plt.figure(figsize=(18, 2*num_images))

for i in range(num_images):
    ax = plt.subplot(3, num_images, i + 1)
    plt.imshow(encoded_imgs[i].reshape(10, 10))
    plt.gray()
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.set_title("Compressed Image")

    ax = plt.subplot(3, num_images, num_images + i + 1)
    plt.imshow(encrypted_imgs[i].reshape(10, 10))
    plt.gray()
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.set_title("Encrypted Image")

plt.tight_layout()
plt.show()
