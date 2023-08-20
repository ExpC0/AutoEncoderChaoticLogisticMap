import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from chaoticLogisticMap import keygen

image_directory = "./img/"

# Parameters for the chaotic map
x_0 = 0.001
r = 3.9159

# List all image files in the directory
image_files = [filename for filename in os.listdir(image_directory) if filename.endswith(".png")]

# Initialize an array to store shuffled images
shuffled_images = []
original_images = []

# Loop through each image file
for filename in image_files:
    img = Image.open(os.path.join(image_directory, filename))
    original_images.append(img)

    # Convert the PIL image to a NumPy array
    img_array = np.array(img)

    # Generate the shuffling matrix using the chaotic map
    size = img_array.shape[0] * img_array.shape[1]
    shuffling_matrix = np.array(keygen(x_0, r, size))

    # Reshaping shuffling matrix by chaotic map
    shuffling_matrix = shuffling_matrix.reshape(img_array.shape[0], img_array.shape[1])

    # Shuffling algorithm
    def shuffle_image(image, matrix):
        shuffled_image = np.copy(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if np.amax(image[i]) >= np.amax(matrix[j]):
                    shuffled_image[i] = np.roll(shuffled_image[i], max(matrix[j]))
                else:
                    shuffled_image[i] = np.roll(shuffled_image[i], -max(matrix[j]))
        return shuffled_image

    # Shuffle the image using the algorithm
    shuffled_img = shuffle_image(img_array, shuffling_matrix)
    np.save(f"shuffled_"+filename, shuffled_img)
    shuffled_images.append(shuffled_img)

# Convert the shuffled_images list to a NumPy array
shuffled_images = np.array(shuffled_images)

# Plot all shuffled images together
num_images = shuffled_images.shape[0]
plt.figure(figsize=(18, 2*num_images))

for i in range(num_images):
    plt.subplot(num_images, 2, 2 * i + 1)
    plt.imshow(original_images[i])
    plt.title("Original Image")
    plt.axis('on')

    plt.subplot(num_images, 2, 2 * i + 2)
    plt.imshow(shuffled_images[i])
    plt.title("Shuffled Image")
    plt.axis('on')

plt.tight_layout()
plt.show()




