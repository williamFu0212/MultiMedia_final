import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import numpy as np

#對比度
def adjust_contrast(img, factor):
    # 轉換為 NumPy 陣列
    img_array = np.array(img)

    # 將陣列轉換為浮點數類型(避免像素值*因子後被侷限在0~255)
    img_array_float = img_array.astype(float)

    # 對每個通道分別進行對比度調整
    adjusted_array = (img_array_float - 128) * factor + 128
    #print(np.max(np.abs(adjusted_array)))
    
    # 計算縮放因子，以確保值在合法範圍內
    scale_factor = 255 / max(np.max(adjusted_array), abs(np.min(adjusted_array)))
    #print(np.max(adjusted_array))
    #print(np.min(adjusted_array))

    # 重新縮放整個數組，並進行 Clip
    adjusted_array = np.clip(adjusted_array * scale_factor, 0, 255)
    #print(np.max(adjusted_array))
    #print(np.min(adjusted_array))

    # 轉換為 uint8 類型
    adjusted_array = adjusted_array.astype(np.uint8)

    # 轉換回 PIL 影像
    adjusted_img = Image.fromarray(adjusted_array)

    return adjusted_img

#飽和度
def adjust_saturation(img, factor):
    # 轉換為 HSV 色彩模型(顏色、飽和度、亮度)
    img_hsv = img.convert("HSV")

    # 提取 H、S、V 三個通道
    h, s, v = img_hsv.split()

    # 飽和度調整
    s = s.point(lambda i: i * factor)

    # 合併調整後的通道
    adjusted_img_hsv = Image.merge("HSV", (h, s, v))

    # 轉換回 RGB 色彩模型
    adjusted_img_rgb = adjusted_img_hsv.convert("RGB")

    return adjusted_img_rgb

#亮度
def adjust_brightness(img, brightness_factor):
    # 轉換為 HSV 色彩模型
    img_hsv = img.convert("HSV")

    # 提取 H、S、V 三個通道
    h, s, v = img_hsv.split()

    # 亮度調整
    v = v.point(lambda i: min(max(i * brightness_factor, 0), 255))

    # 合併調整後的通道
    adjusted_img_hsv = Image.merge("HSV", (h, s, v))

    # 轉換回 RGB 色彩模型
    adjusted_img_rgb = adjusted_img_hsv.convert("RGB")

    return adjusted_img_rgb

#R/G/B亮度
def adjust_channel_brightness(img, channel, brightness_factor):
    # 分割 RGB 通道
    r, g, b = img.split()

    # 將指定通道進行亮度調整
    if channel == 'R':
        r = r.point(lambda i: min(max(i * brightness_factor, 0), 255))
    elif channel == 'G':
        g = g.point(lambda i: min(max(i * brightness_factor, 0), 255))
    elif channel == 'B':
        b = b.point(lambda i: min(max(i * brightness_factor, 0), 255))

    # 合併調整後的通道
    adjusted_img = Image.merge("RGB", (r, g, b))

    return adjusted_img

# 開啟影像
path_detail = "C:\\Users\\User\\Desktop\\python\\Class\\multimedia\\final_project\\"
path_input = "input_pic\\"
path_output = "output_pic\\"
image_name = 'house.jpg'

img = Image.open(path_detail + path_input + image_name)

# 調整對比度    (圖片, 倍率)
contrast_up = adjust_contrast(img, 2)
contrast_down = adjust_contrast(img, 0.5)

# 調整飽和度    (圖片, 倍率)
color_up = adjust_saturation(img, 2)
color_down = adjust_saturation(img, 0.5)

# 調整亮度
bright_up = adjust_brightness(img, 1.25)  # 提高亮度
bright_down = adjust_brightness(img, 0.8)  # 降低亮度

# 顯示結果
plt.figure(figsize=(15, 10))

# 對比度
plt.subplot(231)    #rows, columns, No.
plt.imshow(contrast_up)
plt.title('contrast:2')

plt.subplot(234)
plt.imshow(contrast_down)
plt.title('contrast:0.5')

# 飽和度
plt.subplot(232)
plt.imshow(color_up)
plt.title('color:2')

plt.subplot(235)
plt.imshow(color_down)
plt.title('color:0.5')

# 亮度
plt.subplot(233)
plt.imshow(bright_up)
plt.title('bright:1.25')

plt.subplot(236)
plt.imshow(bright_down)
plt.title('bright:0.8')

#show
plt.show()


# 分別調整R、G、B的亮度
red_up = adjust_channel_brightness(img, 'R', 1.2)    # 提高紅色通道亮度
green_down = adjust_channel_brightness(img, 'G', 1.2)  # 提高綠色通道亮度
blue_up = adjust_channel_brightness(img, 'B', 1.2)     # 提高藍色通道亮度

# 顯示結果
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.imshow(red_up)
plt.title('Red Channel Brightness Up')

plt.subplot(132)
plt.imshow(green_down)
plt.title('Green Channel Brightness Down')

plt.subplot(133)
plt.imshow(blue_up)
plt.title('Blue Channel Brightness Up')

plt.show()
