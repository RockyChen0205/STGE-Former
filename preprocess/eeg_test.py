# 该文件用于处理set文件，并按照
import mne
import matplotlib.pyplot as plt
import os
import numpy

dir_path= 'D:\MDDestimationProject\mywork\data'

def list_files(directory):
    # 获取目录下的所有文件和文件夹名
    files = os.listdir(directory)

    # 打印每个文件名
    for file_name in files:
        print(file_name)

# 获取该目录下所有文件的文件名
# files_list = []
# files_list = list_files(dir_path)


# 测试mne读取.set文件
"""
通过mne.io.read_raw_eeglab来读取.set文件
得到原始数据对象
"""
raw = mne.io.read_raw_eeglab("D:\\MDDestimationProject\\mywork\\data\\02010002_process.set",preload=True)
# raw.plot(start=5, duration=5)
print(type(raw))
print(raw.info)
print(raw.times)
raw.plot()
plt.show()

# 以下部分参考 https://github.com/eeyhsong/EEG-Conformer/











