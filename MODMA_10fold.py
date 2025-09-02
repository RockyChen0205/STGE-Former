import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import TensorDataset, DataLoader, Dataset, Subset
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch
from sklearn import metrics
from glob import glob
import mne
import random
import os
import logging
import argparse
import time
from sklearn.model_selection import KFold
from data_provider.data_loader import MODMA_Dataset
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt



# 导入模型
# from model.model import Model
from model.STGEFormer import Model
# from model.STGEFormer_noembed import Model
# from model.SFormer import Model
# from model.TGEFormer import Model
# from model.STFormer import Model

# 配置日志级别
logging.basicConfig(level=logging.INFO)

# 设置环境变量固定随机数种子
os.environ['PYTHONHASHSEED'] = str(42)
def setup_seed(seed=42):
    '''
    固定随机数种子
    :param seed:随机数种子数值
    :return none
    '''

    # 固定torch的随机数种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 固定numpy的随机数种子
    np.random.seed(seed)

    # 固定python的随机数种子
    random.seed(seed)

    # 保证CUDA的确定性操作
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#* 固定随机数种子
setup_seed()




def read_MODMA_data_path(path, expNo):
    '''
    根据路径读取文件

    :param path: 数据集目录路径
    :param expNo: 实验对象编号
    :return:
    '''
    # 获取该目录所有文件的名称列表
    all_file_path = os.listdir(path)

    # 正，负样本路径
    negative_file_path = []
    negative_file_label_path = []
    positive_file_path = []
    positive_file_label_path = []

    # 遍历所有文件，并获取所有列表
    for i in all_file_path:
        # 负样本标签
        if all_file_path[i][3] == '1' and all_file_path[i][8] == '_':
            negative_file_label_path.append(all_file_path[i])
        # 负样本
        elif all_file_path[i][3] == '1':
            negative_file_path.append(all_file_path[i])
        # 正样本标签
        elif all_file_path[i][8] == '_':
            positive_file_label_path.append(all_file_path[i])
        # 正样本
        else:
            positive_file_path.append(all_file_path[i])

    return negative_file_path, negative_file_label_path, positive_file_path, positive_file_label_path



def extract_subject_id(folder_path):
    '''
    获取文件名的前八个字符，并且不能重复

    :param folder_path: 数据集目录路径
    :return:  返回一个列表
    '''
    file_names = os.listdir(folder_path)
    prefixes = {}
    for file_name in file_names:
        prefix = file_name[:8]
        if prefix not in prefixes:
            prefixes[prefix] = True
    return list(prefixes.keys())






if __name__ == '__main__':



    #* --------定义一些可变参数--------------------------------------------------------
    #* 描述，任务名称：Mywork
    parser = argparse.ArgumentParser(description='Mywork')

    #* model define
    parser.add_argument('--seq_len', type=int, default=501, help='input sequence length')
    parser.add_argument('--enc_in', type=int, default=128, help='encoder input size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    # parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    # 2048 = 4 * 512
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--use_norm', type=bool, default=True, help='use norm and denorm')

    #* optimization
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    #* GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    # store_true 是一个布尔型的标志位，当在命令行中使用--use_multi_gpu时，该参数的值会被设置为True.
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')

    #* STGEFormer
    parser.add_argument('--cls_len', type=int, default=3, help='global token length')
    parser.add_argument('--graph_depth', type=int, default=3, help='graph aggregation depth')
    parser.add_argument('--knn', type=int, default=16, help='graph nearest neighbors')
    parser.add_argument('--embed_dim', type=int, default=16, help='node embed dim')

    # 获取所有命令行参数，以及如果gpu可用，则使用gpu
    args = parser.parse_args()
    print(args)
    #* ========cuda环境=================================================================
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    device = 'cuda'

    # 如果使用多卡
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]



    # epoch和batch size设置
    num_epochs = args.train_epochs
    batch_size = args.batch_size

    #* ========加载数据==============================================
    # 定义数据路径
    data_path = '/tmp/pycharm_project_29/after_process_data'
    subject_list = extract_subject_id(data_path)

    # 遍历subject_list，将其转换为字符串，然后使用逗号和空格将它们连接成一个字符串
    logging.info(f'subject列表:{", ".join([str(item) for item in subject_list])}')

    # 加载所有对象的数据
    data_list = []
    label_list = []
    for subject in subject_list:  # 假设对象编号从1到53
        # 读取数据
        data = np.load(f'{data_path}/{subject}.npy')
        # 读取标签
        label = np.load(f'{data_path}/{subject}_label.npy')
        # 获取数据形状
        # logging.info(f'{subject}的数据形状:{data.shape}')
        # logging.info(f'{subject}的标签形状:{label.shape}')
        # 添加到列表中
        data_list.append(data)
        label_list.append(label)



    # 将所有数据合并
    all_data = np.concatenate(data_list, axis=0)
    all_labels = np.concatenate(label_list, axis=0)

    # 改成数据集
    dataset = MODMA_Dataset(all_data, all_labels)


    #* ======开始十折交叉验证========================================

    # 十折交叉验证
    kf = KFold(n_splits=10, shuffle=True)

    # 用于保存每折的度量值
    fold_metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "specificity": [],
        "train_loss": [],
        "val_loss": []
    }

    # 交叉循环验证
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f'Fold {fold + 1} started')

        # 创建数据加载器
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size, shuffle=True)

        # 初始化模型、损失函数和优化器
        model = Model(args).to(device)
        criterion = nn.CrossEntropyLoss()
        # criterion = nn.MSELoss()
        # 二分类任务，使用BCEloss
        # criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        train_losses = []
        val_losses = []
        #* 训练模型
        num_epochs = args.train_epochs
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                # batch_y = batch_y.to(torch.float).unsqueeze(-1)  # (32, 1)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_losses.append(train_loss / len(train_loader))
            print(f'Epoch {epoch + 1}, Train Loss: {train_losses[-1]}')

        #* 验证模型
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                # labels = labels.to(torch.float).unsqueeze(-1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_losses.append(val_loss / len(val_loader))

        # 计算度量值
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='binary')
        recall = recall_score(all_labels, all_preds, average='binary')

        # 计算混淆矩阵以计算 specificity
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        specificity = tn / (tn + fp)

        print(
            f'Validation Loss: {val_losses[-1]}, Accuracy: {accuracy * 100}%, Precision: {precision}, Recall: {recall}, Specificity: {specificity}')

        # 记录当前 fold 的结果
        fold_metrics["accuracy"].append(accuracy)
        fold_metrics["precision"].append(precision)
        fold_metrics["recall"].append(recall)
        fold_metrics["specificity"].append(specificity)
        fold_metrics["train_loss"].append(train_losses[-1])
        fold_metrics["val_loss"].append(val_losses[-1])

    # 计算并打印平均值
    avg_metrics = {k: np.mean(v) for k, v in fold_metrics.items()}
    print("\nFinal Average Metrics:")
    for metric, value in avg_metrics.items():
        print(f'{metric.capitalize()}: {value}')

    # 保存实验结果
    results_dir = './results/'
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'metrics.txt'), 'w') as f:
        for metric, value in avg_metrics.items():
            f.write(f'{metric.capitalize()}: {value}\n')

    # 绘制并保存 Loss 曲线
    plt.figure()
    plt.plot(range(1, len(fold_metrics['train_loss']) + 1), fold_metrics['train_loss'], label='Train Loss')
    plt.plot(range(1, len(fold_metrics['val_loss']) + 1), fold_metrics['val_loss'], label='Validation Loss')
    plt.xlabel('Fold')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.savefig(os.path.join(results_dir, 'loss_curve.png'))
    plt.show()



























