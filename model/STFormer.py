"""
without graph-enhanced
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


class nconv(nn.Module):
    """
    基于图卷积的操作
    """
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('nwl,vw->nvl', (x, A))
        return x.contiguous()


class mixprop(nn.Module):
    """
    mixprop 类实现了图卷积操作，并通过多层线性变换进行聚合

    通过图卷积操作对输入数据进行处理，并将多层的输出进行拼接后通过线性变换得到最终结果
    """
    def __init__(self, c_in, c_out, gdep, dropout=0.2, alpha=0.1):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = nn.Linear((gdep+1)*c_in, c_out)
        self.gdep = gdep  # 图卷积的深度
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        # normalization to adj
        # 假设adj的形状为[bs node node]
        # sum表示每个节点的度之和（单向图）
        # d的形状为[bs node]
        d = adj.sum(1)  # 计算每个节点的度
        # 这行代码的作用是对邻接矩阵进行归一化处理，使其成为一个概率转移矩阵
        # 它通过对邻接矩阵的每一行除以对应的节点度来实现这一点
        # d.view(-1, 1): 将 d 转换为形状为 (n_nodes, 1) 的列向量
        a = adj / d.view(-1, 1)  # 归一化邻接矩阵

        h = x  # 初始化隐藏状态
        out = [h]  # 存储每一层的输出
        # 图卷积操作
        for _ in range(self.gdep):
            # h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            h = F.dropout(h, self.dropout)
            h = self.nconv(h, a)  # 图卷积操作
            out.append(h)  # 保存当前层的输出
        ho = torch.cat(out, dim=2)  # 将所有层的输出拼接到一起
        ho = self.mlp(ho)  # 线性变换
        return ho


class graph_constructor(nn.Module):
    """
    graph_constructor 类用于构建图的邻接矩阵
    1. 对节点嵌入应用两个线性变换和激活函数。
    2. 计算两个变换后的向量的点积差，并应用 ReLU 激活函数。
    3. 如果需要，保留前 k 大值以减少计算复杂度。
    4. 返回最终的邻接矩阵。
    """
    def __init__(self, nnodes, k, dim, alpha=1, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes  # 节点数量

        self.lin1 = nn.Linear(dim, dim)  # 第一个线性变换层
        self.lin2 = nn.Linear(dim, dim)  # 第二线性变换层

        self.k = k  # 保留的邻居数量
        self.dim = dim  # 特征的维度
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, node_emb):
        nodevec1 = F.gelu(self.alpha*self.lin1(node_emb))  # 应用第一个线性变换和激活函数
        nodevec2 = F.gelu(self.alpha*self.lin2(node_emb))  # 应用第二个线性变换和激活函数
        # 构建邻接矩阵
        # 1. 计算 nodevec1 和 nodevec2 的点积差。
        # 2. 然后，使用 ReLU 激活函数，将所有负值置为零。
        # 3. 结果是一个初步的邻接矩阵 adj，其中每个元素表示节点之间的连接强度
        adj = F.relu(torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0)))
        # 稀疏化操作
        if self.k < node_emb.shape[0]:
            n_nodes = node_emb.shape[0]
            # 创建一个全零矩阵 mask，用于存储稀疏化后的邻接矩阵。
            mask = torch.zeros(n_nodes, n_nodes).to(node_emb.device)  # 创建一个零矩阵作为掩码
            mask.fill_(float('0'))
            # 对邻接矩阵添加一个小随机噪声（0.01）以打破可能的平局
            # 使用 topk 操作保留每行中前 k 大的值及其对应的索引
            s1, t1 = (adj + torch.rand_like(adj)*0.01).topk(self.k, 1)
            # 将保留的值的位置标记为 1
            mask.scatter_(1, t1, s1.fill_(1))
            # 用掩码矩阵 mask 来稀疏化原始的邻接矩阵，只保留前 k 大的值，其他位置置为 0
            adj = adj*mask
        return adj



class GraphEncoder(nn.Module):
    """
    GraphEncoder 类结合了Transformer层和图神经网络层，用于处理输入数据。
    """
    def __init__(self, attn_layers, gnn_layers, gl_layer, node_embs, cls_len, norm_layer=None):
        super(GraphEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)  # Transformers层列表
        self.graph_layers = nn.ModuleList(gnn_layers)  # 图神经网络层列表
        self.graph_learning = gl_layer  # 图学习层
        self.norm = norm_layer  # 归一化层
        self.cls_len = cls_len  # 类别长度
        self.node_embs = node_embs  # 节点嵌入

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []  # 保存注意力权重
        gcls_len = self.cls_len
        # 使用图学习层生成或更新节点之间的邻接矩阵 adj
        adj = self.graph_learning(self.node_embs)

        for i, attn_layer in enumerate(self.attn_layers):
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)

            if i < len(self.graph_layers):
                g = x[:, :gcls_len]  # 提取类别部分
                # 重新排列数据张量的维度，以适应 GNN 的输入格式
                g = rearrange(g, '(b n) p d -> (b p) n d', n=self.node_embs.shape[0])  # 重新排列
                # 通过 GNN 层计算节点之间的特征传递，并与原始特征进行残差连接
                g = self.graph_layers[i](g, adj) + g  # 图神经网络
                g = rearrange(g, '(b p) n d -> (b n) p d', p=gcls_len)  # 重新排列回原样
                x[:, :gcls_len] = g  # 更新列别部分

            if self.norm is not None:
                x = self.norm(x)

        return x, attns

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
        # Encoder
        # self.encoder = GraphEncoder(
        #     [
        #         EncoderLayer(
        #             AttentionLayer(
        #                 FullAttention(False, configs.factor, attention_dropout=configs.dropout,
        #                               output_attention=configs.output_attention), configs.d_model, configs.n_heads),
        #             configs.d_model,
        #             configs.d_ff,
        #             dropout=configs.dropout,
        #             activation=configs.activation
        #         ) for l in range(configs.e_layers)
        #     ],
        #     [mixprop(configs.d_model, configs.d_model, gdep) for _ in range(configs.e_layers-1)],
        #     graph_constructor(configs.enc_in, knn, embed_dim, alpha=gc_alpha),
        #     nn.Parameter(torch.randn(configs.enc_in, embed_dim), requires_grad=True), cls_len,
        #     norm_layer=nn.LayerNorm(configs.d_model)
        # )
        self.temporal_encoder= Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), 512, configs.n_heads),
                    512,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(512)
        )

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
        # self.head_nf = configs.d_model * \
        #     int((configs.seq_len - patch_len) / stride + 2)

        self.classifer_head = nn.Sequential(
            # nn.Flatten(start_dim=-3),
            # nn.Dropout(configs.dropout),
            nn.Linear(
                131072, 128),
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
        #* temporal部分
        # u: [bs * n_vars x patch_num x d_model]
        # enc_out, n_vars = self.patch_embedding(x_enc)
        # ! enc_out: [bs*n_vars, patch_num, patch_len]

        # _, N, _ = x_enc.shape  # B L N
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates
        enc_out=x_enc
   
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.temporal_encoder(enc_out, attn_mask=None)


        #* spatial部分
        # enc_out_spatial = self.enc_embedding(x_enc)
        enc_out_spatial = x_enc
        enc_out_spatial = enc_out_spatial.permute(0, 2, 1)
        enc_out_spatial, attns = self.spatial_encoder(enc_out_spatial)

        # Decoder
        enc_out = torch.reshape(enc_out,(enc_out.shape[0],-1))
        enc_out_spatial = torch.reshape(enc_out_spatial,(enc_out_spatial.shape[0],-1))
        enc_out = torch.cat([enc_out,enc_out_spatial],dim=1)

        # dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        # dec_out = dec_out.permute(0, 2, 1)
        enc_out = self.act(enc_out)
        enc_out = self.dropout(enc_out)
        dec_out = self.classifer_head(enc_out)

        return dec_out



    def forward(self, x_enc):
        dec_out = self.classification(x_enc)
        return dec_out  # [B,]








