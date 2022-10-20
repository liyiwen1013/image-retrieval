
import tensorflow as tf
import matplotlib.pyplot as plt
import os



# 使用数据增强——利用 ImageDataGenerator 来设置数据增强
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.08,
        height_shift_range=0.08,
        shear_range=0.1,
        zoom_range=0.1,
        fill_mode='nearest')

# 显示几个随机增强后的训练图像
from keras.preprocessing import image

data_dir = r'../数据采集/dish/4'
fnames = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir)]
img_path = fnames[15]  # 选择一张图像进行增强
img = image.load_img(img_path, target_size=(150, 150))  # 读取图像并调整大小
x = image.img_to_array(img)   # 将其转换为形状为（150，150，3）的 Numpy 数组
x = x.reshape((1,) + x.shape)  # 将其形状转换为（1，150，150，3）

i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()
