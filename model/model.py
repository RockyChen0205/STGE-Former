import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Encoder import EncoderLayer, Encoder
from model.self_attention import FullAttention, AttentionLayer
from model.embedding import DataEmbedding_inverted
import numpy as np
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
class ClassificationHead(nn.Sequential):
    '''
    分类头，用于将特征向量转换为分类结果
    '''

    def __init__(self, emb_size):
        """
        初始化分类头。
        emb_size: int，特征向量的维度大小。
        n_classes: int，分类结果的类别数量。
        """
        super().__init__()

        # global average pooling
        # 全局平均池化
        # clshead 用于将输入进行池化、标准化和线性映射，以进行最终的分类任务
        # b: batch
        # e: 相当于通道
        # self.clshead = nn.Sequential(
        #     Reduce('b n e -> b e', reduction='mean'),
        #     nn.LayerNorm(emb_size),
        #     nn.Linear(emb_size, n_classes)
        # )
        # fc 是一个前馈神经网络，用于对输入进行多层线性变换和激活函数处理
        # 它主要用于特征提取和压缩
        self.fc = nn.Sequential(
            # 使用平均池化
            Reduce('b n e -> b n', reduction='mean'),
            # nn.LayerNorm(emb_size),
            nn.Linear(128, 32),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 8),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        # 通过view()函数，将张量x的大小调整为(batch_size, -1)，
        # x = x.contiguous().view(x.size(0), -1)
        # print('input of nn.Sigmoid:', out.shape)
        # print(out)
        outlayer = nn.Sigmoid()
        out = outlayer(self.fc(x))
        return out


class Model(nn.Module):

    def __init__(self, configs):
        '''
        初始化模型
        :param configs:
                seq_len: 序列长度
                pred_len: 预测长度
                output_attention: 是否使用注意力机制
                use_norm: 是否使用标准化的配置
                d_model: 模型维度
                embed: 嵌入的配置
                freq: 频率
                class_strategy: 分类策略
                d_ff:
                drop_out: 是否使用drop_out策略
                e_layer



        '''

        # 调用父类初始化方法
        super(Model, self).__init__()

        # 初始化序列长度
        self.seq_len = configs.seq_len
        # 是否使用output_attention
        self.output_attention = configs.output_attention
        # 是否使用标准化的配置
        self.use_norm = configs.use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.dropout)
        # self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        # 初始化编码器的架构
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        # self.classifer = ClassificationHead(configs.d_model)
        # 初始化投影层
        # self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        # 分类投影层
        self.projector = nn.Linear(configs.d_model * configs.enc_in, 2)
    def classification(self, x_enc):
        '''
        分类器的核心功能

        :param x_enc: 需要进行处理的数据
        :return:
        '''
        if self.use_norm:
            # Normalization from Non-stationary Transformer

            # * 从非平稳变换器进行归一化
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            # 对计算得到的方差加上 1e-5，这是一个很小的正数，用来确保后续开方不会遇到数值稳定性问题。
            # 开根号得到标准差
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            # * 将 x_enc 除以其标准差 stdev，使得数据具有单位标准差。
            # 不是不等号！
            x_enc /= stdev

        _, N, _ = x_enc.shape  # B L N
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc)  # covariates (e.g timestamp) can be also embedded as tokens

        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N
        # dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]  # filter the covariates

        # 反归一化，归一化的逆过程
        # 这种反归一化步骤通常在使用归一化数据进行模型训练时是必要的，因为模型学习到的是归一化后的特征，而最终的预测结果需要映射回原始数据的空间。
        # if self.use_norm:
        #     # De-Normalization from Non-stationary Transformer
        #     dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        #     dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        # return dec_out

        # output = self.classifer(enc_out)
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
        output = self.projector(output)  # (batch_size, num_classes)
        # outlayer = nn.Sigmoid()
        # output = outlayer(output)
        return output

    def forward(self, x_enc):
        output = self.classification(x_enc)
        return output

