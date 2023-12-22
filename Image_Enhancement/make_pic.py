import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# def add_noise(img, n):
#     img2 = img.copy()
#     for _ in range(n):
#         x = int(np.random.random() * img.shape[0])
#         y = int(np.random.random() * img.shape[1])
        
#         # 對每個通道添加獨立的噪聲
#         img2[x, y, 0] = 255  # 紅色通道
#         img2[x, y, 1] = 255  # 綠色通道
#         img2[x, y, 2] = 255  # 藍色通道

#     return img2


path_detail = "C:\\Users\\User\\Desktop\\python\\Class\\multimedia\\final_project\\"
path_input = "input_pic\\"
image_names = ['cat.jpg', 'house.jpg', 'shadow.jpg']

for i in range(3) :

    # 读取图像
    path = path_detail + path_input
    image = cv2.imread(path + image_names[i])
    
    # noised_image = add_noise(image, 160000)
    # Image.fromarray(noised_image).save(path + "noised_" + image_names[i])

    # 指定模糊核的大小
    kernel_size = (7, 7)

    # 使用均值滤波进行图像模糊
    blurred_image = cv2.blur(image, kernel_size)

    Image.fromarray(blurred_image.astype(np.uint8)).save(path + "blurred_" + image_names[i])