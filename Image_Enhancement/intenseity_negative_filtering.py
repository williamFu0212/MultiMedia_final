import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# T > 1.0：增強圖像的對比度和亮度。圖像變得更亮
# T < 1.0：降低圖像的對比度和亮度。圖像變得更暗
# T = 1.0：圖像不會發生變化。
def intensity_transformation(image, T):
    # Basic intensity transformation: g(x, y) = T * f(x, y)
    transformed_image = (T * image).astype(np.uint8)
    return transformed_image

def image_negative(f):
    if len(f.shape) == 3 :
        # Split the color channels
        b, g, r = cv2.split(f)

        # Perform image negative on each channel
        neg_b = 255 - b
        neg_g = 255 - g
        neg_r = 255 - r

        # Merge the channels back into a color image
        neg_image = cv2.merge([neg_b, neg_g, neg_r])
    else :
        neg_image = 255 - f
    return neg_image


def main():
    path_detail = "C:\\Users\\User\\Desktop\\python\\Class\\multimedia\\final_project\\"
    path_input = "input_pic\\"
    path_output = "output_pic\\"
    # image_names = ['blurred_cat.jpg', 'blurred_house.jpg', 'blurred_shadow.jpg']
    image_names = ['house.jpg', 'mouse2.jpg', 'skeleton_orig.bmp']
    images_num = 3
    for i in range(images_num) :
        # Load an image
        original_image = cv2.imread(path_detail + path_input + image_names[i])

        # Intensity Transformation
        Ts = [0.7, 0.85, 1.15, 2]
        for j in range(4) :
            intensity_transformed_image = intensity_transformation(original_image, Ts[j])
            plt.subplot(3, 2, j + 3)
            plt.imshow(intensity_transformed_image, cmap='gray')
            plt.title(f'Intensity_transformed_' + str(j))
            Image.fromarray(intensity_transformed_image.astype(np.uint8)).save(path_detail + path_output + "intensity_transformed_" + str(Ts[j]) + image_names[i])
        
        # Image Negetive
        neg_image = image_negative(original_image)

        # save
        output_name = image_names[i]
        
        Image.fromarray(neg_image.astype(np.uint8)).save(path_detail + path_output + "image_negtived_" + output_name)
        
        # 显示原始图像
        plt.subplot(3, 2, 1)
        plt.imshow(original_image, cmap='gray')
        plt.title('Original Image')

        plt.subplot(3, 2, 2)
        plt.imshow(neg_image, cmap='gray')
        plt.title('Negtived Image')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()