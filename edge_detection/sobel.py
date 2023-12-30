import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def apply_sobel(image_path):
    # 定義 Sobel 濾波器的卷積核
    kernel_sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    kernel_sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

    # 使用 filter2D 函數進行卷積操作
    sobel_x = cv2.filter2D(image, cv2.CV_64F, kernel_sobel_x)
    sobel_y = cv2.filter2D(image, cv2.CV_64F, kernel_sobel_y)

    # 結合 x 和 y 方向的濾波結果
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_combined = np.uint8(sobel_combined)

    return sobel_combined

path_detail = "C:\\Users\\User\\Desktop\\python\\Class\\multimedia\\final_project\\"
path_input = "input_pic\\"
path_output = "output_pic\\"
image_name = 'house.jpg'

# 讀取圖片
image = cv2.imread(path_detail + path_input + image_name, cv2.IMREAD_GRAYSCALE)
# 使用函數
edge_image = apply_sobel(image)

output_name = path_detail + path_output + "Sobel_edge_" + image_name
Image.fromarray(edge_image.astype(np.uint8)).save(output_name)
plt.subplot(1, 2, 2)
plt.imshow(edge_image, cmap ='gray')
plt.title('edge_detection')

# 显示原始图像和处理后的图像
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title(f'Original Image')

plt.tight_layout()
plt.show()

