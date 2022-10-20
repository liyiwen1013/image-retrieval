import os
import tensorflow as tf
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片


# 图片预测，传入 模型 和 图片的绝对地址
def predict(model, path):
    test_img = tf.keras.preprocessing.image.load_img(path, target_size=(150, 150))
    test_img = tf.keras.preprocessing.image.img_to_array(test_img)  # 类型转换
    test_img = tf.expand_dims(test_img, 0)  # 扩充一维
    result = model.predict(test_img)  # 预测
    scores = tf.nn.softmax(result[0])  # 得分转换为概率

    print("图片中有：{}个盘子, 概率值为：{:.2%}".format(class_names[np.argmax(scores)], np.max(scores)))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码
    plt.imshow(mpimg.imread(path))
    print(class_names)
    print(class_names[np.argmax(scores)])
    plt.title("图片中有：{}个盘子, 概率值为：{:.2%}".format(class_names[np.argmax(scores)], np.max(scores)))
    plt.axis('off')  # 不显示坐标轴
    plt.show()


# 遍历文件夹的所有文件，返回所有文件的绝对路径列表
def root_path(files_path):
    mp4_path_list = []  # 存放文件的绝对地址
    for root, dirs, files in os.walk(files_path):
        for f in files:  # 添加文件的地址到列表中
            mp4_path_list.append(os.path.join(root, f))

    return mp4_path_list  # 返回所有文件的绝对路径列表


if __name__ == '__main__':
    model = load_model("xiaomu(10类).h5")  # 读取模型
    # predict(model, '13.jpg')  # 单个图片预测
    data_dir = r'../数据采集/dish'
    class_names = os.listdir(data_dir)  # 获取数据集类别（文件夹名）
    path_list = root_path(data_dir + r'/10')  # 文件结构目录（用于整个文件夹图片预测）
    print(path_list)
    for i in path_list:
        predict(model, i)  # 逐个预测



