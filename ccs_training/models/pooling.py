# encoding: utf-8
"""
@author:  andy.ybm
@contact: andy.ybm@alibaba-inc.com
"""

import torch
import torch.nn.functional as F
from torch import nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class GeneralizedMeanPooling(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'


class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    """ Same, but norm is trainable
    """

    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = nn.Parameter(torch.ones(1) * norm)


class AdaptiveAvgMaxPool2d(nn.Module):
    def __init__(self):
        super(AdaptiveAvgMaxPool2d, self).__init__()
        self.avgpool = FastGlobalAvgPool2d()

    def forward(self, x):
        x_avg = self.avgpool(x, self.output_size)
        x_max = F.adaptive_max_pool2d(x, 1)
        x = x_max + x_avg
        return x


class FastGlobalAvgPool2d(nn.Module):
    def __init__(self, flatten=False):
        super(FastGlobalAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, feat_dims, n_head, attn_dropout=0.1, feat_size=8, position_embedding=False):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.position_embedding = position_embedding
        self.feat_dims = feat_dims
        self.n_head = n_head
        self.feat_size = feat_size
        if self.position_embedding:
            ys,xs = torch.meshgrid(torch.arange(feat_size), torch.arange(feat_size))
            xs, ys = xs.flatten(), ys.flatten()
            self.k_gem_pos = nn.Parameter(torch.Tensor(1, feat_size * feat_size + 1, feat_dims))
            self.k_fmap_pos = nn.Parameter(torch.Tensor(2 * feat_size - 1 , 2 * feat_size - 1, feat_dims))
            self.v_gem_pos = nn.Parameter(torch.Tensor(1, feat_size * feat_size + 1, feat_dims))
            self.v_fmap_pos = nn.Parameter(torch.Tensor(2 * feat_size - 1 , 2 * feat_size - 1, feat_dims))
            self.x_index = torch.cat(list(map(lambda x: x - xs + feat_size - 1, xs)))
            self.y_index = torch.cat(list(map(lambda x: x - ys + feat_size - 1, ys)))
            print('use position_embedding')

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if self.position_embedding:
            self.k_pos = torch.cat([torch.cat([self.k_fmap_pos[self.y_index, self.x_index].reshape(self.feat_size**2, self.feat_size**2, -1), self.k_gem_pos[:, 1:, :]], dim=0), self.k_gem_pos.transpose(0, 1)], dim=1)
            self.v_pos = torch.cat([torch.cat([self.v_fmap_pos[self.y_index, self.x_index].reshape(self.feat_size**2, self.feat_size**2, -1), self.v_gem_pos[:, 1:, :]], dim=0), self.v_gem_pos.transpose(0, 1)], dim=1)
            if torch.cuda.is_available():
                self.k_pos = self.k_pos.cuda()
                self.v_pos = self.v_pos.cuda()
            # print(self.k_pos.shape, self.v_pos.shape)
            # print((q / self.temperature).unsqueeze(-2).shape)
            # print(self.k_pos.shape)
            # print(self.k_pos.transpose(-2, -1).shape)
            # a = torch.matmul((q / self.temperature).unsqueeze(-2), self.k_pos.transpose(-2, -1)).squeeze(-2)
            attn += torch.matmul((q / self.temperature).unsqueeze(-2), self.k_pos.transpose(-2, -1)).squeeze(-2)
            # print('attn', a.shape, attn.shape)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        if self.position_embedding:
            # print(attn.unsqueeze(-1).transpose(-1, -2).shape)
            # print(self.v_pos.shape)
            # a = torch.matmul(attn.unsqueeze(-1).transpose(-1, -2), self.v_pos).squeeze(-2)
            output += torch.matmul(attn.unsqueeze(-1).transpose(-1, -2), self.v_pos).squeeze(-2)
            # print('output', a.shape, output.shape)
        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, position_embedding=False):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, feat_dims=d_k, n_head=n_head, position_embedding=position_embedding)

        self.dropout = nn.Dropout(dropout)
        # self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        # q = self.layer_norm(q)
        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        # self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        # x = self.layer_norm(x)

        return x


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, position_embedding=False):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout, position_embedding=position_embedding)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class SelfAttenPooling(nn.Module):
    ''' Compose with two layers '''
    def __init__(self, cfg):
        super(SelfAttenPooling, self).__init__()
        self.cfg = cfg
        n_head = self.cfg.get('n_head', 1)
        d_model = self.cfg.get('feat_dims', 2048)
        d_inner = self.cfg.get('feat_dims', 2048)
        d_k = self.cfg.get('d_k', 2048)
        d_v = self.cfg.get('d_v', 2048)
        dropout = self.cfg.get('dropout', 0.1)
        n_layers = self.cfg.get('n_layers', 3)
        position_embedding = self.cfg.get('position_embedding', False)
        self.base_pooling = GeneralizedMeanPoolingP()
        position_embedding_list = [position_embedding] + [False] * (n_layers - 1)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, position_embedding=position_embedding_list[tt])
            for tt in range(n_layers)])
        print('n_layers: {}, n_head:{}, d_model:{}, d_k:{}, d_v:{}, dropout:{}'.format(n_layers, n_head, d_model, d_k, d_v, dropout))

    def forward(self, x, attn_mask=None):
        batch_size = x.shape[0]
        dims = x.shape[1]
        base_feat = self.base_pooling(x).view(batch_size, 1, dims)
        x = x.permute(0, 2, 3, 1).reshape(batch_size, -1, dims)
        x = torch.cat([base_feat, x], dim=1)
        enc_output = x
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, slf_attn_mask=attn_mask)

        return enc_output[:, 0, :].squeeze()


if __name__ == '__main__':
    cfg = {
        'n_head': 1,
        'd_k': 2048,
        'd_v': 2048,
        'position_embedding': True
    }
    pooling = SelfAttenPooling(cfg)
    x = torch.rand(2, 2048, 8, 8)
    if torch.cuda.is_available():
        pooling = pooling.cuda()
        x = x.cuda()
    out = pooling(x)
    print(out, out.shape)

