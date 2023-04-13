import torch.nn as nn
import torch

from .transformer import Transformer


class attention(nn.Module):
    """
    Class for the attention module of the model

    Methods:
    --------
        forward: torch.Tensor -> torch.Tensor
            forward pass of the attention module

    """

    def __init__(self):
        super(attention, self).__init__()
        self.att_dim = 256
        self.W = nn.Parameter(torch.randn(256, self.att_dim))
        self.V = nn.Parameter(torch.randn(self.att_dim, 1))
        self.scale = self.att_dim**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        e = torch.matmul(x, self.W)
        e = torch.matmul(torch.tanh(e), self.V)
        e = e * self.scale
        n1 = torch.exp(e)
        n2 = torch.sum(torch.exp(e), 1, keepdim=True)
        alpha = torch.div(n1, n2)
        x = torch.sum(torch.mul(alpha, x), 1)
        return x


class encoder(nn.Module):
    """
    Class for the encoder of the model that contains time encoder

    Methods:
    --------
        forward: torch.Tensor -> torch.Tensor
            forward pass of the encoder

    """

    def __init__(self, input_channels):
        super(encoder, self).__init__()
        self.base_module = BaseNet(input_channels=input_channels)
        self.attention_module = attention()

    def forward(self, x: torch.Tensor):
        x = self.base_module(x)
        x = self.attention_module(x)
        return x

class projection_head(nn.Module):
    def __init__(self, num_class, input_dim: int = 256):
        super(projection_head, self).__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_class, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection_head(x)
        return x

class resnetWithAttention(nn.Module):
    def __init__(self, input_channels, epoch_len, num_class, do_context):
        super(resnetWithAttention, self).__init__()

        self.epoch_len = epoch_len
        self.do_context = do_context
        self.base_encoder = encoder(input_channels)
        self.tfmr = Transformer(256, 4, 4, 256, dropout=0.1)
        self.projection_head = projection_head(num_class)
        
    def forward(self, x):
        if not self.do_context:
            x = self.base_encoder(x[:, (self.epoch_len // 2), :, :])
            x = self.projection_head(x)
        else:
            surr = []
            for i in range(self.epoch_len):
                surr.append(self.base_encoder(x[:, i, :, :])) # 128, 1, 3000
            surr = torch.stack(surr, dim=1)
            x = self.tfmr(surr)
            x = self.projection_head(x)
        return x




# Convolution Function
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes,
                     out_planes,
                     kernel_size=7,
                     stride=stride,
                     padding=3,
                     bias=False)


# Basic Building Block
class BasicBlock_Bottle(nn.Module):
    expansion = 4

    def __init__(self, inplanes3, planes, stride=1, downsample=None):
        super(BasicBlock_Bottle, self).__init__()
        self.conv1 = nn.Conv1d(inplanes3, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv1d(planes,
                               planes,
                               kernel_size=25,
                               stride=stride,
                               padding=12,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride
        self.conv3 = nn.Conv1d(planes,
                               self.expansion * planes,
                               kernel_size=1,
                               bias=False)  #
        self.bn3 = nn.BatchNorm1d(self.expansion * planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# Main 1D-RESNET Model
class BaseNet(nn.Module):

    def __init__(self, input_channels=1, layers=[3, 4, 6, 3]):
        self.inplanes3 = 16

        super(BaseNet, self).__init__()

        self.conv1 = nn.Conv1d(input_channels,
                               16,
                               kernel_size=71,
                               stride=2,
                               padding=35,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=71, stride=2, padding=35)

        self.layer3x3_1 = self._make_layer3(BasicBlock_Bottle,
                                            8,
                                            layers[0],
                                            stride=1)
        self.layer3x3_2 = self._make_layer3(BasicBlock_Bottle,
                                            16,
                                            layers[1],
                                            stride=2)
        self.layer3x3_3 = self._make_layer3(BasicBlock_Bottle,
                                            32,
                                            layers[2],
                                            stride=2)
        self.layer3x3_4 = self._make_layer3(BasicBlock_Bottle,
                                            64,
                                            layers[3],
                                            stride=2)

    def _make_layer3(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes3 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.inplanes3,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes3, planes, stride, downsample))
        self.inplanes3 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes3, planes))

        return nn.Sequential(*layers)

    def forward(self, x0):
        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)

        x = self.layer3x3_1(x0)
        x = self.layer3x3_2(x)
        x = self.layer3x3_3(x)
        x = self.layer3x3_4(x)
        return x
