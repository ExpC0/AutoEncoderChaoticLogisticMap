import numpy as np
import matplotlib.pyplot as plt
from chaoticLogisticMap import keygen


decoded_images = []


for i in range(3):
    decoded_img = np.load(f"decoded_img_{i}.npy")
    decoded_img = decoded_img.reshape(256, 256, 3)
    decoded_images.append(decoded_img)


# Parameters for the chaotic map (use the same values as before)
x_0 = 0.001
r =3.9159

# Generate the shuffling matrix using the chaotic map
shuffling_matrix = np.array(keygen(x_0, r, 256 * 256))
shuffling_matrix = shuffling_matrix.reshape(256, 256)

# Deshuffling algorithm
def deshuffle_image(shuffled_img, matrix):
    deshuffled_image = np.copy(shuffled_img)
    for i in range(shuffled_img.shape[0]):
        for j in range(shuffled_img.shape[1]):
            if np.amax(shuffled_img[i]) >= np.amax(matrix[j]) :
                deshuffled_image[i] = np.roll(deshuffled_image[i],-max(matrix[j]))
            else:
                deshuffled_image[i] = np.roll(deshuffled_image[i],max(matrix[j]))
    return deshuffled_image


deshuffled_images = []
for i in range(len(decoded_images)):
    deshuffled_img = deshuffle_image(decoded_images[i], shuffling_matrix)
    deshuffled_images.append(deshuffled_img)

# Display the shuffled and deshuffled images
num_images = len(decoded_images)
plt.figure(figsize=(18, 2*num_images))

for i in range(num_images):
    plt.subplot(num_images, 2, 2 * i + 1)
    plt.imshow(decoded_images[i])
    plt.title("Decoded (Shuffled) Image")
    plt.axis('on')

    plt.subplot(num_images, 2, 2 * i + 2)
    plt.imshow(deshuffled_images[i])
    plt.title("Deshuffled Image")
    plt.axis('on')

plt.tight_layout()
plt.show()



# #*********************************<><><>*********************************************
# #This code is Normal Deshuffler , it takes input as shuffled img and give output as deshuffled img
# #it works perfectly fine , so shuffle and deshuffle algorithm work fine
# #so if still can't obtain orignal image after passing through shuffle,encoder,encrypt,decrypt,decoder then
# #there is loss for compression by autoencoder
#
# import os
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# from chaoticLogisticMap import keygen
#
# # Directory where the shuffled images are located
# shuffled_images_directory = "./"
#
# # Parameters for the chaotic map
# x_0 = 0.001
# r = 3.9159
#
# # List all shuffled image files in the directory
# shuffled_image_files = [filename for filename in os.listdir(shuffled_images_directory) if filename.startswith("shuffled_img")]
#
# # Initialize arrays to store original and deshuffled images
# shuffled_images = []
# deshuffled_images = []
#
# # Loop through each shuffled image file
# for filename in shuffled_image_files:
#     # Load the shuffled image using numpy
#     shuffled_img = np.load(os.path.join(shuffled_images_directory, filename))
#     shuffled_images.append(shuffled_img)
#
#
#     # Generate the shuffling matrix using the chaotic map
#     size = shuffled_img.shape[0] * shuffled_img.shape[1]
#     shuffling_matrix = np.array(keygen(x_0, r, size))
#
#     # Reshaping shuffling matrix by chaotic map
#     shuffling_matrix = shuffling_matrix.reshape(shuffled_img.shape[0], shuffled_img.shape[1])
#
#     # Deshuffling algorithm
#     def deshuffle_image(shuffled_img, matrix):
#         deshuffled_image = np.copy(shuffled_img)
#         for i in range(shuffled_img.shape[0]):
#             for j in range(shuffled_img.shape[1]):
#                 if np.amax(shuffled_img[i]) >= np.amax(matrix[j]):
#                     deshuffled_image[i] = np.roll(deshuffled_image[i], -max(matrix[j]))
#                 else:
#                     deshuffled_image[i] = np.roll(deshuffled_image[i], max(matrix[j]))
#         return deshuffled_image
#
#     # Deshuffle the image using the algorithm
#     deshuffled_img = deshuffle_image(shuffled_img, shuffling_matrix)
#
#     deshuffled_images.append(deshuffled_img)
#
# # Convert the arrays to numpy arrays
# shuffled_images = np.array(shuffled_images)
# deshuffled_images = np.array(deshuffled_images)
#
# # Plot original and deshuffled images together
# num_images = shuffled_images.shape[0]
# plt.figure(figsize=(15, 2*num_images))
#
# for i in range(num_images):
#     plt.subplot(num_images, 2, 2 * i + 1)
#     plt.imshow(shuffled_images[i])
#     plt.title("shuffled Image")
#     plt.axis('on')
#
#     plt.subplot(num_images, 2, 2 * i + 2)
#     plt.imshow(deshuffled_images[i])
#     plt.title("Deshuffled Image")
#     plt.axis('on')
#
# plt.tight_layout()
# plt.show()








