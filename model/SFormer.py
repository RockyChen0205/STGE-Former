"""
without T-steam
"""

import torch
from torch import nn
import torch.nn.functional as F
from model.Encoder import EncoderLayer, Encoder
from model.self_attention import FullAttention, AttentionLayer
from model.embedding import PatchEmbedding, DataEmbedding_inverted
from einops import repeat
from einops import rearrange
import numpy as np


class Flatten_Head(nn.Module):
    '''
    这个类用于将输入数据进行展平、线性变换和dropout处理
    '''
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        # 使用 nn.Flatten(start_dim=-2) 来创建一个展平层，它会将从倒数第二个维度开始的所有维度展平成一维。
        self.flatten = nn.Flatten(start_dim=-2)

        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x



class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf

    1. 补丁嵌入:
        - 将输入数据切分为patch，并进行嵌入。
        - 添加类别标记。
    2. 编码器:
        - 应用 Transformer 编码器和图神经网络层。
        - 重塑输出。
    3. 预测头:
        - 应用特定任务的预测头。
        - 返回最终输出。
    4. 任务处理:
        - 根据任务名称选择不同的方法进行处理。
        - 返回相应任务的输出。
    """

    def __init__(self, configs, patch_len=128, stride=64, gc_alpha=1):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        # 序列长度
        self.seq_len = configs.seq_len
        padding = stride
        # 类别长度
        cls_len = configs.cls_len
        # 图卷积深度
        gdep = configs.graph_depth
        # k近邻
        knn = configs.knn
        embed_dim = configs.embed_dim

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.dropout)
        # global tokens
        # 形状为 (1, cls_len, configs.d_model)
        # nn.Parameter 是 PyTorch 中的一个类，用于创建可学习的参数。这意味着在模型训练过程中，这些参数会被优化
        # 用于捕获序列的信息
        self.cls_token = nn.Parameter(torch.randn(1, cls_len, 128))

        self.value_embedding = nn.Linear(501, 512)
        
        # only spatial_encoder
        self.spatial_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.enc_in, 64),
                    configs.enc_in,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.enc_in)
        )

        # Prediction Head
        # 这玩意就是patch_len * patch_num
        self.head_nf = configs.d_model * \
            int((configs.seq_len - patch_len) / stride + 2)

        self.classifer_head = nn.Sequential(
            # nn.Flatten(start_dim=-3),
            # nn.Dropout(configs.dropout),
            nn.Linear(
                65536, 128),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2)
        )
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)

    def classification(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding

        # x_enc = x_enc.permute(0, 2, 1)
        # ! x_enc: [bs, n_vars, seq_len]
        x_enc = self.value_embedding(x_enc)
        #* 无temporal部分


        #* spatial部分
        # enc_out_spatial = self.enc_embedding(x_enc)
        enc_out_spatial = x_enc
        enc_out_spatial = enc_out_spatial.permute(0, 2, 1)
        enc_out_spatial, attns = self.spatial_encoder(enc_out_spatial)

        # Decoder
        # enc_out = torch.reshape(enc_out,(enc_out.shape[0],-1))
        enc_out_spatial = torch.reshape(enc_out_spatial,(enc_out_spatial.shape[0],-1))
        enc_out = enc_out_spatial
        # enc_out = torch.cat([enc_out,enc_out_spatial],dim=1)

        # dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        # dec_out = dec_out.permute(0, 2, 1)
        enc_out = self.act(enc_out)
        enc_out = self.dropout(enc_out)
        dec_out = self.classifer_head(enc_out)

        return dec_out



    def forward(self, x_enc):
        dec_out = self.classification(x_enc)
        return dec_out  # [B,]








