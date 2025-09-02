import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        位置嵌入层，用于为序列中的每个位置提供唯一的向量表示。

        通过将位置信息嵌入到模型中，模型能够学习到输入序列中词的相对位置和顺序信息。
        本层通过预计算并存储位置编码（positional encodings），以在前向传播时添加到输入张量上，
        从而让模型能够区分输入序列中不同位置的词。

        参数:
        - d_model: 模型的维度，即每个位置嵌入向量的大小。
        - max_len: 支持的最大序列长度，默认为5000。
        """
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        # 计算并存储位置编码，这些编码在模型训练过程中不被更新（require_grad=False）
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        # 为每个位置创建一个张量，范围从0到max_len。
        position = torch.arange(0, max_len).float().unsqueeze(1)
        # 计算用于编码的因子，这些因子基于位置和模型维度计算得出
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        # 使用正弦和余弦函数对位置编码进行填充，偶数列使用正弦，奇数列使用余弦。
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 将位置编码张量重塑为(1, max_len, d_model)，以便能够与输入序列相加。
        pe = pe.unsqueeze(0)
        # 将位置编码注册为缓冲区，这样就可以在前向传播时自动使用。
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 从self.pe中切片获取列，列数与x的第二个维度大小相同
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    '''
    这个类的主要目的是实现一种嵌入层，用于处理输入数据并将其转换为固定大小的向量表示
    '''
    def __init__(self, c_in, d_model):
        '''
        参数：
        c_in: 输入特征的数量
        d_model: 输出嵌入向量的维度
        '''
        super(TokenEmbedding, self).__init__()
        # 根据 PyTorch 的版本选择填充（padding）的值。如果版本大于等于 1.5.0，则设置为 1；否则设置为 2。这是因为不同版本的 PyTorch 对于循环填充（circular padding）的行为有所不同。
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        # 创建一个一维卷积层
        # padding_mode： 使用循环填充模式
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        # 遍历模块中的所有子模块
        for m in self.modules():
            # 如果子模块是nn.Conv1d类型，则对其权重进行Kaiming正态分布初始化。这通常用于带有Leaky ReLU激活函数的网络层
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # 将输入张量 x 的维度进行调整，从 [batch_size, seq_len, c_in] 变为 [batch_size, c_in, seq_len]
        # 卷积后，再次调整输出张量的维度，从 [batch_size, d_model, seq_len] 变为 [batch_size, seq_len, d_model]。
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class FixedEmbedding(nn.Module):
    '''
    这个类的主要目的是实现一种固定的嵌入层，用于处理输入数据并将其转换为固定大小的向量表示
    '''
    def __init__(self, c_in, d_model):
        '''
        c_in: 输入特征的数量。
        d_model: 输出嵌入向量的维度
        '''
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        # unsqueeze()将一维向量扩展为二维，形状变为(c_in, 1)
        position = torch.arange(0, c_in).float().unsqueeze(1)

        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    '''
    总结来说，这段代码实现了一个时间嵌入层，可以根据不同的时间特征（小时、工作日、日、月、分钟）将输入的时间特征转换为固定大小的向量表示
    这些嵌入向量可以用于增强时间序列模型的性能，特别是在处理具有周期性模式的数据时。
    '''
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        # 不同时间特征的大小
        # 这里分为不同类型的时间特征
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        # 创建不同时间特征的嵌入层
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        # 如果存在分钟嵌入层 minute_embed，则使用它处理输入张量 x 的第 4 维特征，并存储结果为 minute_x
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        # 类似地，处理其不同维度的特征，并将结果相加。
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    '''
    这个类的主要目的是实现一种时间特征嵌入层，用于处理时间序列数据中的时间特征
    并将其转换为固定大小的向量表示。

    总结来说，这段代码实现了一个时间特征嵌入层，它通过一个线性层将输入的时间特征转换为固定大小的向量表示
    这种方法简单直接，适用于多种时间频率的数据
    输入的时间特征应该已经被编码为数值形式，然后通过线性层进行转换
    这种方式相比于使用嵌入层（如 nn.Embedding 或自定义的 FixedEmbedding）更加灵活，因为它可以处理连续的数值特征，而不是离散的类别特征。
    '''

    def __init__(self, d_model, embed_type='timeF', freq='h'):
        '''
        d_model: 输出嵌入向量的维度。
        embed_type: 嵌入类型，默认为 'timeF'。
        freq: 时间频率，默认为 'h'（小时）。
        '''
        super(TimeFeatureEmbedding, self).__init__()
        # 定义一个字典 freq_map，用于映射不同的时间频率到对应的特征维度：
        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        # 根据 freq 的值从 freq_map 字典中获取对应的特征维度 d_inp。
        d_inp = freq_map[freq]
        # 创建一个线性层 embed，输入大小为 d_inp，输出大小为 d_model，并且不使用偏置项。
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        '''
        c_in: 输入特征的数量。
        d_model: 输出嵌入向量的维度。
        embed_type: 嵌入类型，默认为 'fixed'。
        freq: 时间频率，默认为 'h'（小时）。
        dropout: Dropout 层的比例，默认为 0.1。
        '''
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        '''
        x: 主要的时间序列数据
        x_mark: 可选的时间标记数据
        '''
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        # c_in == seq_len
        self.value_embedding = nn.Linear(c_in, 512)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # 将输入张量 x 的维度进行调整，从 [batch_size, seq_len, c_in] 变为 [batch_size, c_in, seq_len]。
        # x = x.permute(0, 2, 1)
        #! 这里将代码注释掉，因为我输入的数据本来就是 [batch_size c_in seq_len]

        # x的形状为[batch_size c_in seq_len]
        # x: [Batch Variate Time]

        # x = self.value_embedding(x)

        # x: [Batch Variate d_model]
        return self.dropout(x)

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        # ReplicationPad1d用于在一维数据上进行复制填充（replication padding）
        # 它会沿着指定的维度在数据的两侧添加填充值，填充值是数据边界值的重复
        # 这个元组 (0, padding) 指定了填充的位置和数量： 0表示在左侧不进行填充， padding表示在右侧填充padding 个值
        # self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        # patch_len是patch的长度，d_model是输出的维度
        # !  batch_size * channel * pathch_len  --> batch_size * channel * d_model
        self.value_embedding = nn.Linear(501, 512)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # ! x: batch_size * channel * seq_len
        # do patching
        # 获取通道数 n_var
        n_vars = x.shape[1]
        # ! x: batch_size * channel * (seq_len+padding)
        # x = self.padding_patch_layer(x)
        # x=self.value_embedding(x)
        # print('value_embedding: ', x.shape)
        # 然后使用unfold方法将张量沿最后一个维度切分成多个patch，每个patch的长度为self.patch_len，步长为self.stride。这通常用于将一维序列转换为适合卷积或其他局部感知操作的形式。
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # 使用torch.reshape方法重新排列张量的形状，使其变为(batch_size * channel, patch数量, patch_len)，这样可以更容易地应用后续的编码步骤。
        # ! (batch_size * channel, patch数量, patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = x + self.position_embedding(x)
        

        return self.dropout(x), n_vars



class ST_Embedding(nn.Module):
    """
    用于创建Spatial-Temporal Embedding
    """
    def __init__(self, seq_len=501, channel=128, temp_len=512, spatial_len=128, dropout=0.1):
        """
        Spatial-Temporal Collaborative Embedding

        输入数据形状为[batch_size, channel, seq_len]
        :param seq_len: 每个通道的序列长度
        :param channel: 通道数
        :param temp_len: 映射到temporal这一维的长度
        :param spatial_len: 映射到spatial这一维的长度
        :param dropout: dropout的比例
        """
        super(ST_Embedding, self).__init__()
        # c_in == seq_len
        # 将输入的序列长度seq_len映射到指定的时间维度长度temp_len
        self.temporal_embedding = nn.Linear(seq_len, temp_len)
        # 将输入的通道数channel映射到指定的空间维度长度spatial_len
        self.spatial_embedding = nn.Linear(channel, spatial_len)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x=self.temporal_embedding(x)
        #! [b, channel, temp_len]
        x=x.transpose(1,2)
        #! [b, temp_len, channel]
        x=self.spatial_embedding(x)
        print('shape of x after spatial embedding transform:', x.shape)
        #! [b, temp_len, spatial_len]
        x=x.transpose(1,2)
        #! [b, spatial_len, temp_len]
        print('shape of x after collaborative embedding:', x.shape)


