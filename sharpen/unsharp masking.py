import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 自定义模糊掩蔽函数
def blur(image, size):
    image_height, image_width = image.shape
    blurred_image = np.zeros_like(image)

    for i in range(image_height):
        for j in range(image_width):
            window = image[max(0, i - size//2):min(image_height, i + size//2 + 1),
                           max(0, j - size//2):min(image_width, j + size//2 + 1)]
            blurred_image[i, j] = np.mean(window)

    return blurred_image

# 读取图像
path_detail = "C:\\Users\\User\\Desktop\\python\\Class\\multimedia\\final_project\\"
path_input = "input_pic\\"
path_output = "output_pic\\"
image_paths = [path_detail + path_input + 'skeleton_orig.bmp', path_detail + path_input + 'blurry_moon.tif']
output_paths = [path_detail + path_output + 'skeleton_blur_sharpened.bmp', path_detail + path_output + 'blurry_moon_blur_sharpened.tif']

plt.figure(figsize=(10, 5))

for i, image_path in enumerate(image_paths):
    image = np.array(Image.open(image_path).convert('L'))  # 转为灰度

    # 使用模糊掩蔽进行图像锐化
    blurred_image = blur(image, 5)  # 这里使用大小为 5 的模糊掩蔽

    # 銳化影像
    sharpened_image = 2 * image - blurred_image

    # 将处理后的图像保存
    output_path = output_paths[i]
    Image.fromarray(sharpened_image.astype(np.uint8)).save(output_path)

    # 显示原始图像和处理后的图像
    plt.subplot(2, 2, i*2+1)
    plt.imshow(image, cmap='gray')
    plt.title(f'Original Image {i+1}')

    plt.subplot(2, 2, i*2+2)
    plt.imshow(sharpened_image, cmap='gray')
    plt.title(f'Sharpened Image {i+1}')

plt.tight_layout()
plt.show()


