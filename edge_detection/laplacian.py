import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def apply_laplacian(image_path):
    kernel_laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    laplacian = cv2.filter2D(image, cv2.CV_64F, kernel_laplacian)
    laplacian = np.uint8(np.absolute(laplacian))
    return laplacian


path_detail = "C:\\Users\\User\\Desktop\\python\\Class\\multimedia\\final_project\\"
path_input = "input_pic\\"
path_output = "output_pic\\"
image_name = 'house.jpg'

# 讀取圖片
image = cv2.imread(path_detail + path_input + image_name, cv2.IMREAD_GRAYSCALE)
# 使用函數
edge_image = apply_laplacian(image)

output_name = path_detail + path_output + "Lap_edge_" + image_name
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
