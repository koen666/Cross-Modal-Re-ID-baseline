# check.py=>检查npy文件数据是否正常
import numpy as np

# 例子：查看 train_rgb_resized_img.npy
arr = np.load('./data/SYSU-MM01/train_rgb_resized_img.npy')

print("形状:", arr.shape)      # 看看有多少数据
print("数据类型:", arr.dtype)  # 看看数据类型
print("前几个数据:", arr[:5])  # 取前5个看看
