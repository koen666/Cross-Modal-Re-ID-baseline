import numpy as np
from PIL import Image
import os

data_path = './data/SYSU-MM01'

# 图片尺寸
fix_image_width = 144
fix_image_height = 288

# RGB 和 IR 的子目录
rgb_dirs = ['rgb_modify/bounding_box_train', 'rgb_modify/query', 'rgb_modify/bounding_box_test']
ir_dirs  = ['ir_modify/bounding_box_train', 'ir_modify/query', 'ir_modify/bounding_box_test']

# 读取图片函数
def read_imgs(file_list):
    train_img = []
    train_label = []
    for idx, img_path in enumerate(file_list):
        img = Image.open(img_path).convert('RGB')
        # 兼容新版 Pillow 的 resample
        resample_method = getattr(Image, 'Resampling', Image).LANCZOS
        img = img.resize((fix_image_width, fix_image_height), resample=resample_method)
        train_img.append(np.array(img))
        train_label.append(idx)  # 顺序编号
    return np.array(train_img), np.array(train_label)

# 扫描 RGB
files_rgb = []
for dir in rgb_dirs:
    img_dir = os.path.join(data_path, dir)
    if os.path.isdir(img_dir):
        new_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith('.jpg')])
        files_rgb.extend(new_files)
print(f"Found {len(files_rgb)} RGB images")

# 扫描 IR
files_ir = []
for dir in ir_dirs:
    img_dir = os.path.join(data_path, dir)
    if os.path.isdir(img_dir):
        new_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith('.jpg')])
        files_ir.extend(new_files)
print(f"Found {len(files_ir)} IR images")

# 读取并保存 RGB
train_rgb_img, train_rgb_label = read_imgs(files_rgb)
np.save(os.path.join(data_path, 'train_rgb_resized_img.npy'), train_rgb_img)
np.save(os.path.join(data_path, 'train_rgb_resized_label.npy'), train_rgb_label)

# 读取并保存 IR
train_ir_img, train_ir_label = read_imgs(files_ir)
np.save(os.path.join(data_path, 'train_ir_resized_img.npy'), train_ir_img)
np.save(os.path.join(data_path, 'train_ir_resized_label.npy'), train_ir_label)

print("Preprocessing completed successfully!")
