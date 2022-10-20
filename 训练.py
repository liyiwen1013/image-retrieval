import keras
from keras.preprocessing import sequence

import PIL
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os


# 将文件夹中的数据加载到tf.data.Dataset中，且加载的同时会打乱数据。
import tensorflow as tf

data_dir = r'../数据采集/dish'
train_ds = tf.keras.preprocessing.image_dataset_from_directory(directory=data_dir,
                                                               batch_size=128,
                                                               validation_split=0.2,
                                                               subset='training',
                                                               seed=666,
                                                               image_size=(150, 150))
val_ds = tf.keras.preprocessing.image_dataset_from_directory(directory=data_dir,
                                                             batch_size=32,
                                                             validation_split=0.2,
                                                             subset='validation',
                                                             seed=666,
                                                             image_size=(150, 150))
class_names = train_ds.class_names   # 获取数据集类别
# print(class_names)

# directory: 数据所在目录，batch_size: 数据批次的大小（默认值）：32，image_size: 图片的统一缩放 150*150，
# batch_size: 数据批次的大小（默认值：32），subset: "training"或"validation"之一。仅在设置validation_split时使用。



# 缓存数据

AUTOTUNE = tf.data.AUTOTUNE  # tf.data用于数据集的构建与预处理
# Dataset.shuffle(buffer_size) 随机打乱、Dataset.prefetch() 并行处理
train_ds = train_ds.cache().shuffle(200).prefetch(buffer_size=AUTOTUNE)
# 当模型在训练的时候，prefetch 使数据集在后台取得 batch。
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 数据增强
from tensorflow.keras import layers
data_argumentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip('horizontal', input_shape=(150, 150, 3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
])

# 构建模型
from tensorflow.keras import layers, models

model = models.Sequential([
    data_argumentation,  # 数据增强
    layers.experimental.preprocessing.Rescaling(1./255),  # 归一化
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPool2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPool2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPool2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPool2D(),
    layers.Conv2D(256, 3, padding='same', activation='relu'),
    layers.MaxPool2D(),
    layers.Dropout(0.2),  # 丢掉一些神经元，处理过拟合
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(len(class_names))  # 模型类别，识别填入
])
# 查看特征图的维度如何随着每层变化
model.summary()

# 编译模型——配置模型用于训练
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam', metrics=['acc'])

# 在监督学习中我们使用梯度下降法时，学习率是一个很重要的指标，因为学习率决定了学习进程的快慢（也可以看作步幅的大小）。
# 如果学习率过大，很可能会越过最优值，反而如果学习率过小，优化的效率可能很低，导致过长的运算时间，所以学习率对于算法性能的表现十分重要。
# 而优化器keras.optimizers.Adam()是解决这个问题的一个方案。其大概的思想是开始的学习率设置为一个较大的值，然后根据次数的增多，动态的减小学习率，以实现效率和效果的兼得。


epochs = 25  # 迭代次数

# 训练模型
# 使用.fit方法进行测试数据与模型的拟合
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# 保存模型
model.save("lyw_plate.h5")

# 绘制训练过程中的损失曲线和精度曲线
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'SimHei'   # 使图形中的中文正常编码显示
plt.rcParams['axes.unicode_minus'] = False   # 使坐标轴刻度表签正常显示正负号

train_acc = history.history['acc']  # 获取训练精度值
val_acc = history.history['val_acc']  # 获取验证精度值
train_loss = history.history['loss']  # 获取训练损失值
val_loss = history.history['val_loss']  # 获取验证损失值

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)  # 画布
ranges = range(epochs)  # x轴
plt.plot(ranges, train_acc, label='训练精度值')
plt.plot(ranges, val_acc, label='验证精度值')
plt.title("训练验证精度值")
plt.xlabel("迭代次数")
plt.ylabel("精度值")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(ranges, train_loss, label='训练损失值')
plt.plot(ranges, val_loss, label='验证损失值')
plt.title("训练验证损失值")
plt.xlabel("迭代次数")
plt.ylabel("损失值")
plt.legend()
plt.show()

