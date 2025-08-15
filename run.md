# 如何运行AGW框架进行跨模态行人重识别

AGW (Adaptive Graph-Wavelet Network)是一个用于跨模态行人重识别的基线框架，支持在RegDB和SYSU-MM01数据集上进行训练和测试。下面我将详细讲解如何让这个框架跑起来。

## 1. 环境准备

首先需要确保你的系统满足以下要求：
- Python 3.x
- PyTorch (建议1.0+版本)
- 支持CUDA的GPU

可以通过以下命令安装必要的Python包：
```
pip install torch torchvision numpy scipy
```

## 2. 数据集准备

### SYSU-MM01数据集
1. 从[官方网站](http://isee.sysu.edu.cn/project/RGBIRReID.htm)下载SYSU-MM01数据集
2. 运行预处理脚本：
```
python pre_process_sysu.py
```
这将把训练数据转换为.npy格式

### RegDB数据集
1. 从[DBPerson-Recog-DB1网站](http://dm.dongguk.edu/link.html)提交版权申请表获取数据集
2. 或者发送邮件至mangye16@gmail.com请求私人下载链接

## 3. 训练模型

### 在SYSU-MM01数据集上训练
```
python train.py --dataset sysu --lr 0.1 --method agw --gpu 1
```

### 在RegDB数据集上训练
```
python train.py --dataset regdb --lr 0.1 --method agw --gpu 1
```

参数说明：
- `--dataset`: 选择数据集("sysu"或"regdb")
- `--lr`: 初始学习率(默认0.1)
- `--method`: 使用的方法("agw"或baseline)
- `--gpu`: 指定使用的GPU编号

### 训练细节
- 采样策略：每个batch随机选择N个行人ID，然后从每个ID中随机选择4张可见光图像和4张热红外图像
- 训练日志保存在`log/`目录下
- 模型保存在`save_model/`目录下

## 4. 测试模型

### 测试SYSU-MM01数据集
```
python test.py --mode all --resume 'model_path' --gpu 1 --dataset sysu
```

### 测试RegDB数据集
```
python test.py --mode all --resume 'model_path' --gpu 1 --dataset regdb --trial 1
```

参数说明：
- `--dataset`: 选择数据集("sysu"或"regdb")
- `--mode`: 对于SYSU-MM01可选择"all"或"indoor"搜索模式
- `--trial`: 对于RegDB数据集指定测试trial
- `--resume`: 指定保存的模型路径
- `--gpu`: 指定使用的GPU编号

## 5. 预期结果

使用预训练的ImageNet模型，预期可以达到以下性能：

### RegDB数据集
- Rank-1准确率: ~70.05%
- mAP: ~66.37%
- mINP: ~50.19%

### SYSU-MM01数据集
- Rank-1准确率: ~47.50%
- mAP: ~47.65%
- mINP: ~35.30%

## 6. 注意事项

1. 由于随机分割的原因，结果可能会有波动
2. 通过调整超参数可能会获得更好的结果
3. 需要手动定义数据路径
4. 如果需要使用预训练模型，可以从Google Drive下载SYSU-MM01的模型

如需更多帮助，可以联系作者邮箱：mangye16@gmail.com