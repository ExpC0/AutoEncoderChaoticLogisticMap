import os
import numpy as np
from matplotlib import pyplot as plt

import chaoticLogisticMap as CLmap

# Load the encrypted images
encrypted_imgs = []

for filename in os.listdir("./"):
    if filename.startswith("encrypted_img"):
        encrypted_img = np.load("./" + filename)
        encrypted_imgs.append(encrypted_img)

encrypted_imgs = np.array(encrypted_imgs)

# Get the size of the encrypted images
size = encrypted_imgs[0].shape[0]

# Generate the chaotic sequence for the key
key = CLmap.keygen(0.971, 3.9159, size)

# Convert key to a NumPy array and then to np.uint8 data type
key = np.array(key, dtype=np.uint8)

# Decryption Function
decrypted_imgs = []

for encrypted_img in encrypted_imgs:
    # Perform bitwise XOR operation again to decrypt
    decrypted_img = np.bitwise_xor(encrypted_img, key)
    decrypted_imgs.append(decrypted_img)

# Save the decrypted images
for i, decrypted_img in enumerate(decrypted_imgs):
    filename = f"decrypted_img_{i}.npy"
    np.save(filename, decrypted_img)

print("Decryption completed and images saved.")

# Display images
num_images = 3
plt.figure(figsize=(18, 2*num_images))


for i in range(num_images):
    ax = plt.subplot(3, num_images, i + 1)
    plt.imshow(encrypted_imgs[i].reshape(10, 10))
    plt.gray()
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.set_title("Encrypted Image")

    ax = plt.subplot(3, num_images, num_images + i + 1)
    plt.imshow(decrypted_imgs[i].reshape(10,10))
    plt.gray()
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.set_title("Decrypted Image")

plt.tight_layout()
plt.show()
