import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
 
def gamma_correction(f, gamma=2.0):
    # Create a copy of the input image
    g = f.copy()

    # Normalize the image to the range [0, 1]
    g_normalized = g / 255.0

    # Apply gamma correction to each pixel
    g_corrected = np.round(255.0 * g_normalized ** gamma).astype(np.uint8)

    return g_corrected

def main():
    path_detail = "C:\\Users\\User\\Desktop\\python\\Class\\multimedia\\final_project\\"
    path_input = "input_pic\\"
    path_output = "output_pic\\"
    image_names = ['skeleton_orig.bmp']

    for i in range(1) :
        # Load an image
        original_image = cv2.imread(path_detail + path_input + image_names[i])

        # gamma_correction
        # 當 gamma = 1 時表示強度不變
        # gamma < 1 會使影像變亮，可改變曝光不足
        # gamma > 1 會使影像變暗，可改善過曝現象。
        Gs = [0.1, 0.5, 1.5, 2]
        for j in range(4) :
            gamma_correction_image = gamma_correction(original_image, Gs[j])
            # save 
            output_name = path_detail + path_output + "gamma_correction_" + str(Gs[j]) + "_" + image_names[i]
            Image.fromarray(gamma_correction_image.astype(np.uint8)).save(output_name)
            plt.subplot(3, 5, 5 * i + 2 + j)
            plt.imshow(gamma_correction_image, cmap='gray')
            plt.title('Gamma_correction, G = ' + str(Gs[j]))

        # 显示原始图像和处理后的图像
        plt.subplot(3, 5, 5 * i + 1)
        plt.imshow(original_image, cmap='gray')
        plt.title(f'Original Image')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
