# 时频谱图分类实验报告

## 实验目的
本次大作业的目标是训练一个卷积神经网络，将24个指令语音的时频谱图进行分类。

## 实验流程
1. 读取数据
   使用`pathlib`包将读取训练数据所在路径
2. 创建数据集
   用`keras.image_dataset_from_directory`创建`batch_size=32`的图像数据集，并做好训练、测试划分
   使用缓存机制优化训练过程
3. 数据预处理
   预处理层包括：
   - `resizing_and_rescaling_layer`：将图像缩放为512*512像素并像素值变换为$[0.,1.]$区间中的值
   - `data_augmentation`：数据增强层，包括随机翻转、随机旋转、随机缩放和随机加对比度
4. 创建和编译模型
   使用`keras.applications`中封装的`ResNet50V2`模型
5. 训练模型并保存
   训练了200个世代，训练过程中的准确度、损失函数变换保存在`./Accuracy_Loss.png`，模型参数保存在`./my_keras_model.h5`
6. 加载模型
   使用`keras.models.load_model`将`./my_keras_model.h5`中的参数加载入新模型
7. 测试训练结果
   在给定的测试数据上评估模型，预测结果写入了`./result.txt`，总体准确度为50.91%（写在文本文档的末尾）

其中1至5部分的代码在`./classify.py`中，6和7部分的代码在`./test.py`中。