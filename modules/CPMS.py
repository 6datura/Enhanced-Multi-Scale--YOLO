import torch
import torch.nn as nn
from torch.nn import functional as F
 
 
class MSCAAttention(nn.Module):
    # SegNext NeurIPS 2022
    # https://github.com/Visual-Attention-Network/SegNeXt/tree/main
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim,  5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
 
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
 
        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)
 
    def forward(self, x):

 
 
class CPMS(nn.Module):
    #BY https://blog.csdn.net/m0_63774211?type=lately
    def __init__(self, channels):
        super().__init__()
 
        self.ca = MSCAAttention(channels)
 
        self.dconv5_5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.dconv1_7 = nn.Conv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels)
        self.dconv7_1 = nn.Conv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels)
        self.dconv1_11 = nn.Conv2d(channels, channels, kernel_size=(1, 11), padding=(0, 5), groups=channels)
        self.dconv11_1 = nn.Conv2d(channels, channels, kernel_size=(11, 1), padding=(5, 0), groups=channels)
        self.dconv1_21 = nn.Conv2d(channels, channels, kernel_size=(1, 21), padding=(0, 10), groups=channels)
        self.dconv21_1 = nn.Conv2d(channels, channels, kernel_size=(21, 1), padding=(10, 0), groups=channels)
        self.conv = nn.Conv2d(channels, channels, kernel_size=(1, 1), padding=0)
        self.act = nn.GELU()
 
    def forward(self, inputs):
        #   Global Perceptron

