import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def self_histogram_equalization(channel):
    # 將通道數組展平為1D
    flat_channel = channel.flatten()

    # 手動計算直方圖
    hist = np.zeros(256, dtype=int)
    for pixel_value in flat_channel:
        hist[pixel_value] += 1

    # 計算累積分佈函數 (CDF)
    cdf = np.cumsum(hist)
    cdf_normalized = cdf / cdf[-1]

    # 應用直方圖均衡化
    equalized_channel = (cdf_normalized[flat_channel] * 255).reshape(channel.shape).astype(np.uint8)

    return equalized_channel


def histogram_processing(image):
    # 如果圖像是彩色圖片，將其轉換為灰度圖像
    if len(image.shape) == 3:
        # 分離顏色通道
        b, g, r = cv2.split(image)

        # 對每個通道應用直方圖均衡化
        b_equalized = self_histogram_equalization(b)
        g_equalized = self_histogram_equalization(g)
        r_equalized = self_histogram_equalization(r)

        # 合併均衡化的通道成為一個彩色圖像
        equalized_image = cv2.merge([b_equalized, g_equalized, r_equalized])
    else:
        # 如果圖像是灰度圖像，直接應用直方圖均衡化
        equalized_image = self_histogram_equalization(image)

    return equalized_image

def main():
    path_detail = "C:\\Users\\User\\Desktop\\python\\Class\\multimedia\\final_project\\"
    path_input = "input_pic\\"
    path_output = "output_pic\\"
    # image_names = ['blurred_cat.jpg', 'blurred_house.jpg', 'blurred_shadow.jpg']
    image_names = ['mouse1.jpg', 'mouse2.jpg', 'skeleton_orig.bmp']

    for i in range(3) :
        # 載入一張圖片
        original_image = cv2.imread(path_detail + path_input + image_names[i])

        # 圖像直方圖處理
        equalized_image = histogram_processing(original_image)
        # 儲存結果
        output_name = image_names[i]
        Image.fromarray(equalized_image.astype(np.uint8)).save(path_detail + path_output + "equalized_" + output_name)

        # 顯示原始圖像和處理後的圖像
        plt.subplot(3, 2, 2 * i + 1)
        plt.imshow(original_image, cmap='gray')
        plt.title('Original Image')

        plt.subplot(3, 2, 2 * i + 2)
        plt.imshow(equalized_image, cmap='gray')
        plt.title('Equalized Image')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
