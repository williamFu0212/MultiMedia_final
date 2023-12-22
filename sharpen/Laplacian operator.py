import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

laplacian_mask = np.array([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]])

def convolve(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    result = np.zeros_like(image)
    for i in range(image_height):
        for j in range(image_width):
            result[i, j] = np.sum(padded_image[i:i+kernel_height, j:j+kernel_width] * kernel)

    return result
path_detail = "C:\\Users\\User\\Desktop\\python\\Class\\multimedia\\final_project\\"
path_input = "input_pic\\"
path_output = "output_pic\\"
# image_paths = [path_detail + path_input + 'skeleton_orig.bmp', path_detail + path_input + 'blurry_moon.tif']
# output_paths = [path_detail + path_output + 'skeleton_laplacian_sharpened.bmp', path_detail + path_output + 'blurry_moon_laplacian_sharpened.tif']
image_paths = [path_detail + path_input + 'mouse1.jpg', path_detail + path_input + 'mouse2.jpg']
output_paths = [path_detail + path_output + 'mouse1_sharpened.jpg', path_detail + path_output + 'mouse2_sharpened.jpg']


plt.figure(figsize=(10, 5))

for i, image_path in enumerate(image_paths):
    image = np.array(Image.open(image_path).convert('L')) 

    laplacian_output = convolve(image, laplacian_mask)
    sharpened_image = image - laplacian_output

    output_path = output_paths[i]
    Image.fromarray(sharpened_image.astype(np.uint8)).save(output_path)

    plt.subplot(2, 2, i*2+1)
    plt.imshow(image, cmap='gray')
    plt.title(f'Original Image {i+1}')

    plt.subplot(2, 2, i*2+2)
    plt.imshow(sharpened_image, cmap='gray')
    plt.title(f'Sharpened Image {i+1}')

plt.tight_layout()
plt.show()
