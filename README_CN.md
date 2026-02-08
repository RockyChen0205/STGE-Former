# STGE-Former: 基于时空图增强Transformer的EEG抑郁症检测

![框架图](assets/framework.svg)

## 概述

本仓库实现了**STGE-Former**（Spatial-Temporal Graph-Enhanced Transformer，时空图增强Transformer），这是一个用于从EEG信号中检测重度抑郁症（MDD）的创新深度学习框架。我们的模型在MODMA数据集上实现了最先进的性能，通过有效捕获EEG数据中的空间和时间依赖关系。

## 模型架构

STGE-Former的整体框架主要包括四个组件：

1. **输入管道**：预处理EEG信号并准备特征提取
2. **空间注意力流**：使用图注意力机制建模不同脑区之间的功能连接
3. **时间图增强注意力流**：在增强的Transformer架构下捕获不同时间间隔内的脑活动相关性
4. **分类头**：整合时空特征进行最终的抑郁症分类

如图所示，这些组件协同工作，提取和融合空间和时间特征，以实现最佳的抑郁症分类性能。

## 安装

### Python环境
- Python 3.8+
- PyTorch 1.10+
- CUDA 11.3+（用于GPU加速）

### MATLAB/EEGLAB环境（可选）
用于EEG数据预处理的MATLAB脚本：
- MATLAB 2023
- EEGLAB 2023.1

### 设置
```bash
git clone https://github.com/RockyChen0205/STGE-Former.git
cd STGE-Former
pip install -r requirements.txt
```

## 数据集准备

### MODMA数据集
1. 从官方来源下载MODMA数据集
2. 将原始EEG文件放置在`data/`目录
3. 运行预处理脚本：

```bash
cd preprocess
python eeg_slice.py
```

### 兰州大学2015 128通道数据集
针对兰州大学128通道静息态EEG数据集：

1. **配置数据路径** 在`preprocess/eeg_preprocess_batch.m`中：
   - 设置`data_path`为原始`.mat` EEG文件目录
   - 设置`output_path`为处理后`.set`文件的输出目录
   - 电极定位文件`preprocess/mychan`已包含在仓库中

2. **运行MATLAB预处理**：
   - 打开MATLAB 2023并启动EEGLAB 2023.1
   - 运行`preprocess/eeg_preprocess_batch.m`
   - 脚本将执行以下步骤：
     - 导入原始EEG数据（128通道，256Hz）
     - 应用0.1-40Hz带通滤波
     - 执行ICA分解（18个成分）
     - 使用ICLabel进行成分分类
     - 移除伪迹成分（脑电、肌电、眼动 > 90%）
     - 应用平均参考
     - 保存处理后的`.set`文件

3. **转换为numpy格式** 用于模型训练：
```bash
cd preprocess
python eeg_slice.py --input /path/to/processed_set_files
```

### 预处理数据
预处理管道生成：
- `.set`格式的时序片段
- 对应的标签文件
- `after_process_data/`中的处理后的numpy数组

## 使用方法

### 训练
在MODMA数据集上运行10折交叉验证：

```bash
# STGE-Former
bash STGEFormer_MODMA_10fold.sh

# 其他模型
bash SFormer_MODMA_10fold.sh
bash STFormer_MODMA_10fold.sh
bash TGEFormer_MODMA_10fold.sh
```

### 评估
指标自动计算并保存到`results/metrics.txt`：
- 准确率（Accuracy）
- 精确率（Precision）
- 召回率（Recall）
- F1分数（F1-Score）
- AUC-ROC

## 项目结构

```
STGE-Former/
├── data/                 # 原始EEG数据文件
├── after_process_data/   # 处理后的numpy数组
├── data_provider/        # 数据加载工具
├── model/               # 模型架构
│   ├── STGEFormer.py    # 主模型实现
│   ├── Encoder.py       # Transformer编码器
│   ├── embedding.py     # 特征嵌入层
│   └── self_attention.py # 注意力机制
├── preprocess/          # 数据预处理脚本
│   ├── eeg_preprocess_batch.m  # MATLAB/EEGLAB批量处理
│   ├── eeg_slice.py     # Python EEG切片
│   ├── mychan           # EEG电极定位文件
│   └── run.sh           # Shell脚本辅助工具
├── utils/               # 工具函数
├── results/             # 训练结果和指标
├── assets/              # 框架图
└── README.md
```

## 许可证

本项目基于MIT许可证开源。
