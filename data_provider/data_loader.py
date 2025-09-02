import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
import scipy.io as sio




# 设置警告过滤器来忽略所有类型的警告。这意味着在程序运行过程中，即使出现了警告信息，也不会显示出来。
# warnings.filterwarnings('ignore')


class MODMA_Dataset(torch.utils.data.Dataset):
    '''
    将MODMA数据集加载为PyTorch的Dataset类

    :paramdata  : 数据
    :labels    : 标签
    '''
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 将数据转换为 PyTorch 张量
        return self.data[idx], self.labels[idx]









