import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
path_detail = "C:\\Users\\User\\Desktop\\python\\Class\\multimedia\\final_project\\input_pic\\"
# a = cv2.imread(path_detail + "skeleton_orig.bmp", cv2.IMREAD_GRAYSCALE)
# b = cv2.imread(path_detail + "blurry_moon.tif", cv2.IMREAD_GRAYSCALE)
a = cv2.imread(path_detail + "mouse1.jpg", cv2.IMREAD_GRAYSCALE)
b = cv2.imread(path_detail + "mouse2.jpg", cv2.IMREAD_GRAYSCALE)


# 定义高提升滤波的参数和滤波器
A1 = 1.5
A2 = 2
filter1 = np.array([[-1, -1, -1],
                    [-1, 8, -1],
                    [-1, -1, -1]])

# 自定义高提升滤波的函数
def apply_filter(image, filter):
    filtered_image = np.zeros_like(image, dtype=np.float64)
    padded_image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            filtered_image[i, j] = np.sum(filter * padded_image[i : i + 3, j : j + 3])

    image = np.clip(filtered_image, 0, 255)
    uint8_image = image.astype(np.uint8)
    return uint8_image

# 对图像进行高提升滤波
firsta = apply_filter(a, filter1)
firstb = apply_filter(b, filter1)

# 根据给定的比例因子对图像进行缩放
aa = cv2.convertScaleAbs(a, alpha=A1)
bb = cv2.convertScaleAbs(b, alpha=A1)
aaa = cv2.convertScaleAbs(a, alpha=A2)
bbb = cv2.convertScaleAbs(b, alpha=A2)

# 获取高提升滤波后的图像
seconda = cv2.add(aaa, firsta)
secondb = cv2.add(bbb, firstb)
firsta = cv2.add(aa, firsta)
firstb = cv2.add(bb, firstb)

# 显示处理结果
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(a, cmap="gray")
plt.title("Original moon")

plt.subplot(2, 3, 2)
plt.imshow(firsta, cmap="gray")
plt.title("A=1.5 moon")

plt.subplot(2, 3, 3)
plt.imshow(seconda, cmap="gray")
plt.title("A=2 moon")

plt.subplot(2, 3, 4)
plt.imshow(b, cmap="gray")
plt.title("Original skeleton")

plt.subplot(2, 3, 5)
plt.imshow(firstb, cmap="gray")
plt.title("A=1.5 skeleton")

plt.subplot(2, 3, 6)
plt.imshow(secondb, cmap="gray")
plt.title("A=2 skeleton")

plt.savefig("spatial_hb.png")
plt.show()
