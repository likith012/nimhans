import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from copy import deepcopy
import math


class GELU(nn.Module):

    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        x = F.gelu(x)
        return x


class SELayer(nn.Module):
    """ Squeeze and Excitation layer
    """
    def __init__(self, channel, reduction_ratio = 16):
        super(SELayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction_ratio, bias = False),
            nn.ReLU(),
            nn.Linear(channel // reduction_ratio, channel, bias = False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class SEBlock(nn.Module):
    """ SE Block architecture"""

    expansion = 1

    def __init__(self, inplanes, planes, stride = 1, downsample = None, reduction_ratio = 16):
        super(SEBlock, self).__init__()

        self.conv1 = nn.Conv1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction_ratio)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        return out

class MRCNN(nn.Module):
    """ Multi-Resolution CNN architecture"""
    def __init__(self, afr_reduced_cnn_size, input_features = 1):
        super(MRCNN, self).__init__()

        drop_rate = 0.5
        self.GELU = GELU()

        self.features1 = nn.Sequential(
            nn.Conv1d(input_features, 64, kernel_size = 50, stride = 6, padding = 24, bias = False),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size = 8, stride = 2, padding = 4),
            nn.Dropout(drop_rate),

            nn.Conv1d(64, 128, kernel_size = 8, stride = 1, padding = 4, bias = False),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.Conv1d(128, 128, kernel_size = 8, stride = 1, padding = 4, bias = False),
            nn.BatchNorm1d(128),
            self.GELU,
            nn.MaxPool1d(kernel_size = 4, stride = 4, padding = 2)
        )

        self.features2 = nn.Sequential(
            nn.Conv1d(input_features, 64, kernel_size = 400, stride = 50, padding = 200, bias = False),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size =4, stride = 2, padding = 2),
            nn.Dropout(drop_rate),

            nn.Conv1d(64, 128, kernel_size = 7, stride = 1, padding = 3, bias = False),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.Conv1d(128, 128, kernel_size = 7, stride = 1, bias = False, padding = 3),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(kernel_size = 2, stride = 2, padding = 1)
        )

        self.dropout = nn.Dropout(drop_rate)
        self.inplanes = 128 # No. of channels in last layer of MRCNN
        self.AFR = self._make_layer(SEBlock, afr_reduced_cnn_size, 1) # Only 1 SEBlock

    def _make_layer(self, block, planes, num_block, stride = 1):

        downsample = None
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size = 1, stride = 1, bias = False),
                nn.BatchNorm1d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, num_block):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x)
        x_concat = torch.cat((x1, x2), dim = 2)
        x_concat = self.dropout(x_concat)
        x_concat = self.AFR(x_concat)

        return x_concat


def attention(query_vector, key_vector, value_vector, dropout = None):
    """ Dot product attention"""
    key_dim = key_vector.size(-1)
    scores =  torch.matmul(query_vector, key_vector.transpose(-2, -1)) / math.sqrt(key_dim)
    scores = F.softmax(scores, dim = -1)

    if dropout is not None:
        scores = dropout(scores)
    return torch.matmul(scores, value_vector), scores

class CausalConv1d(nn.Conv1d):
    """ Causal convolution"""
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride = 1,
        dilation = 1,
        groups = 1,
        bias = True
    ):

        self.__padding = (kernel_size -1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = self.__padding,
            dilation = dilation,
            groups= groups,
            bias = bias
        )

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:,:, : -self.__padding]
        return result

def clones(module, N):
    """ Produces N identical clones of module"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MHA(nn.Module):

    def __init__(self, num_heads, dim, afr_reduced_cnn_size, dropout = 0.1):
        super(MHA, self).__init__()
        assert dim % num_heads == 0
        self.head_dim = dim // num_heads
        self.num_heads = num_heads 

        self.convs = clones(CausalConv1d(afr_reduced_cnn_size, afr_reduced_cnn_size, kernel_size = 7, stride = 1), 3)
        self.linear = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, query, key, value):
        num_batch = query.size(0)

        query_vector = self.convs[0](query).view(num_batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key_vector = self.convs[1](key).view(num_batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value_vector = self.convs[2](value).view(num_batch, -1, self.num_heads, self.head_dim).transpose(1, 2)

        x, self.attn_scores = attention(query_vector, key_vector, value_vector, dropout = self.dropout)
        x = x.transpose(1, 2).contiguous().view(num_batch, -1, self.num_heads * self.head_dim)

        return self.linear(x)

class LayerNorm(nn.Module):
    """ Applying Layer Normalization across the features(-1)"""
    def __init__(self, features, eps = 1e-6):
        super(LayerNorm, self).__init__()
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim = True)
        std = x.std(-1, keepdim = True)
        return self.a2 * (x- mean) / (std + self.eps) + self.b2


class FeedForward(nn.Module):
    """ Feed-forward architecture"""
    def __init__(self, dim_model, dim_expansion, dropout = 0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(dim_model, dim_expansion)
        self.fc2 = nn.Linear(dim_expansion, dim_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

class Encoder(nn.Module):
    """ Encoder architecture which contains both attention and feed-forward modules"""
    def __init__(self, feature_size, self_attn, feed_forward, afr_reduced_cnn_size, dropout):
        super(Encoder, self).__init__()
        self.self_attn = self_attn # Initialized with dim_model, num_heads
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(p = dropout)
        self.feature_size = feature_size

        self.norm = clones(LayerNorm(feature_size), 2)
        
    def forward(self,x):
        attn = self.self_attn(x, x, x)
        x = self.dropout(self.norm[0](attn + x))

        forward = self.feed_forward(x)
        out = self.dropout(self.norm[1](forward + x))

        return out


class AttnSleep(nn.Module):
    """ Combining all the modules"""
    def __init__(self, n_channels=1, n_classes=5, tce_clones=2):
        super(AttnSleep, self).__init__()

        N = tce_clones # Number of TCE clones
        self.input_features = n_channels
        d_model = 80
        d_ff  = 120
        h = 5
        dropout = 0.1
        afr_reduced_cnn_size = 30

        self.mrcnn = MRCNN(afr_reduced_cnn_size, input_features = self.input_features)
        attn = MHA(h, d_model, afr_reduced_cnn_size)
        ff = FeedForward(d_model, d_ff)
        self.encoder_layer = Encoder(d_model, deepcopy(attn), deepcopy(ff), afr_reduced_cnn_size, dropout)

        self.layers = clones(self.encoder_layer, N)
        self.fc = nn.Linear(d_model * afr_reduced_cnn_size, n_classes)
        
    def forward(self, x):
        x = self.mrcnn(x)
        for layer in self.layers:
            x = layer(x)
        B = x.shape[0]
        x = x.contiguous().view(B, -1)
        out = self.fc(x)
        return out