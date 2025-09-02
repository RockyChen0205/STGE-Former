# 读取EEG数据并进行预处理

import mne
import os
import numpy as np

dir_path= 'D:\\MDDestimationProject\\mywork\\data'
files_list = os.listdir(dir_path)

save_path = 'D:\\MDDestimationProject\\mywork\\after_process_data'


# 53个对象
for i in range(53):
    one_subject = []
    one_subject_label = []

    # 读取EEG数据，并进行查看
    raw = mne.io.read_raw_eeglab(f"{dir_path}/{files_list[i]}", preload=True)
    print(raw.info)

    # 由于EEGlab预处理没有去除参考电极Cz,所以这里需要去除Cz通道
    # 检查 Cz 通道是否存在
    if 'Cz' in raw.info['ch_names']:
        # 删除 Cz 通道
        raw.drop_channels(['Cz'])
    else:
        print("Cz 通道不存在")

    # 查看删除后的通道信息
    print(raw.info['ch_names'])


    # 定义段长（2 秒）和段数
    segment_duration = 2.0  # 秒4332
    events = mne.make_fixed_length_events(raw, start=0, duration=segment_duration, stop=raw.times[-1])

    # 划分为 2 秒的 Epochs
    epochs = mne.Epochs(raw, events, tmin=0, tmax=segment_duration, baseline=None, preload=True)

    # 提取 Epochs 数据
    one_subject = epochs.get_data()

    # 打印数据形状
    print(one_subject.shape)  # 例如，(n_epochs, n_channels, n_times)
    # 打印数据类型
    print(type(one_subject))

    # 如果文件名第4个字母是1，则是负样本，否则是正样本
    # 负样本是0，正样本是1
    if files_list[i][3]=='1':
        one_subject_label = [0 for _ in range(len(one_subject))]
    else:
        one_subject_label = [1 for _ in range(len(one_subject))]

    # 将其转换为ndarray, 方便后续转换为.npy文件
    one_subject = np.array(one_subject)
    one_subject_label = np.array(one_subject_label)

    print(f'Shape of {files_list[i][:8]}:')
    print('one_subject:', one_subject.shape)  # (n_epochs, n_channels, n_times)
    print('one_subject_label:', one_subject_label.shape)  # (n_epochs)

    np.save(save_path+f'/{files_list[i][:8]}.npy', one_subject)
    np.save(save_path+f'/{files_list[i][:8]}_label.npy', one_subject_label)

    print(f'{files_list[i]} is finished')
    print('-------------------------finished------------------------')

print('Data preprocess is finished')

