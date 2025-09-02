import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    '''
    这个类的主要目的是实现一个卷积层，用于处理一维时间序列数据
    它首先使用循环填充的一维卷积层对输入的时间序列数据进行卷积操作，接着通过批量归一化和激活函数来增加模型的非线性表达能力，最后通过最大池化层降低输出的维度
    这种结构有助于提取时间序列数据中的局部特征，并通过池化操作减少计算量和过拟合的风险。
    '''
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        # 输入通道数为 c_in，输出通道数也为 c_in，卷积核大小为 3，使用循环填充（padding_mode='circular'）来保持输出的长度不变。
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        # 创建一个一维批量归一化层 norm，输入通道数为 c_in。
        self.norm = nn.BatchNorm1d(c_in)
        # 创建一个 ELU 激活函数层 activation。
        self.activation = nn.ELU()
        # 创建一个一维最大池化层 maxPool，池化窗口大小为 3，步长为 2，填充为 1。
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        # 将张量 x 的维度再次调整，从 [batch_size, c_in, seq_len] 变回 [batch_size, seq_len, c_in]
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    '''
    attention: 注意力机制模块。
    d_model: 模型的隐藏层维度。
    d_ff: 前馈网络的中间层维度，默认为 4 * d_model。
    dropout: Dropout 层的比例，默认为 0.1。
    activation: 激活函数，默认为 "relu"
    '''
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        # d_model指的是每个通道对应的序列长度
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        '''
        - x: 输入张量
        - attn_mask: 注意力掩码，默认为 None
        - tau: 注意力机制的参数，默认为 None
        - delta: 注意力机制的参数，默认为 None
        '''
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        # 将原始输入张量 x 与经过 Dropout 处理的 new_x 相加，这是残差连接的一部分。
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        # 初始化自注意力层列表，支持处理多头自注意力
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        """
        前向传播函数，处理输入数据x，通过注意力层和卷积层（如果存在）进行处理。

        参数:
        - x: 输入数据，形状为 [B, L, D]，其中 B 是批量大小，L 是序列长度，D 是特征维度。
        - attn_mask: 注意力掩码，用于在注意力计算中忽略某些位置。
        - tau: 用于控制softmax温度的参数，可以用于调整注意力分布的锐化程度。
        - delta: 用于在第一层注意力层中添加到输入x的增量，以调整输入数据。

        返回:
        - x: 经过处理后的数据，形状同输入数据x。
        - attns: 注意力权重列表，每个元素对应一个注意力层的输出。
        """
        # x [B, L, D]
        # 初始化一个空列表来存储各注意力层的注意力权重
        attns = []
        # 检查是否具有卷积层，如果有，则通过注意力层和卷积层处理输入
        if self.conv_layers is not None:
            # 遍历注意力层和卷积层的组合
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                # 第一层使用传入的delta值，后续才能不是用delta
                delta = delta if i == 0 else None
                # 通过注意力层处理输入x，得到新的x和注意力权重attn
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                # 通过卷积层处理输入x
                x = conv_layer(x)
                attns.append(attn)
            # 通过卷积层处理输入x
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            #  将最后一层的注意力权重添加到列表中
            attns.append(attn)
        else:
            # 遍历所有的注意力层
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                # 将注意力权重attn添加到列表中
                # print('创建了注意力层')
                attns.append(attn)
        # 如果存在归一化层，对输出x进行归一化处理
        if self.norm is not None:
            x = self.norm(x)

        return x, attns


