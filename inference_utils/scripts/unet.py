import torch
import torch.nn as nn


#----------------------Input Block---------------------------
class DoubleConv(nn.Module):
    '''
    This block does 3*3 convolution twice.
    '''
    def __init__(self, 
                 in_channels,
                 out_channels):
        super().__init__()
        self.block = nn.Sequential(
            
            nn.Conv2d(in_channels, 
                      out_channels, 
                      kernel_size=3, 
                      padding=1,
                      bias=False),
            
            nn.BatchNorm2d(out_channels),
            
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels,
                      out_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            
            nn.BatchNorm2d(out_channels),
            
            nn.ReLU(inplace=True)  
        )
    
    def forward(self, x):
        return self.block(x)


#-----------------------Downsampling----------------------    
class DownBlock(nn.Module):
    '''
    This block is combination of MaxPool2d and DoubleConv.
    It represents one downsampling block.
    '''
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,
                stride=2),
            DoubleConv(in_channels,
                       out_channels)
        )
        
    def forward(self, x):
        return self.down(x)
    
    
#----------------------Bottleneck---------------------------
    
class ResidualBlock(nn.Module):
    def __init__(self,
                 channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels,
                      channels,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels,
                      channels,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
            return self.block(x)
        
class Bottleneck(nn.Module):
    '''
    It contains two residual blocks
    '''
    def __init__(self, 
                 channels):
        super().__init__()
        self.block = nn.Sequential(
            ResidualBlock(channels),
            ResidualBlock(channels)
        )
        
    def forward(self, x):
        return self.block(x)
    
    
#------------------------Attention Gate-----------------

class AttentionGate(nn.Module):
    '''
    Input:
        e: encoder feature
        d: decoder feature
    Output:
        x: concatenated feature 
    '''
    
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.one_one = nn.Sequential(
            nn.Conv2d(
                in_channels,
                128,
                kernel_size=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(128)   
        )
                
        self.imp = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(128,
                      out_channels//2,
                      kernel_size=1,
                      padding=0,
                      bias=True),
            nn.Sigmoid()
        )
    
    def forward(self, 
                e: torch.Tensor,
                d: torch.Tensor):
        
        e_dash = self.one_one(e)
        d_dash = self.one_one(d)
        sum = e_dash + d_dash
        imp_score = self.imp(sum)
        e = e * imp_score
        concat_feature = torch.concat([e, d], dim=1) #dim=1 as we want to concat along channel in B,C,H,W
        return concat_feature
    
#------------------------UpSampling------------------


class UpBlock(nn.Module):
    '''
    Upsamples the downsampled input. 
    Inlcudes the attention gate logic.
    '''
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels,
                               out_channels,
                               kernel_size=2,
                               stride=2,
                               padding=0,
                               bias=False)
        self.attn = AttentionGate(out_channels, 2*out_channels)
        self.double_conv = DoubleConv(2*out_channels, out_channels)
                    
    def forward(self, e, d):
        d = self.conv_transpose(d)
        d = self.double_conv(self.attn(e, d))
        return d
    
#-----------------------------U Net---------------------------------------
class UNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.input_block = DoubleConv(1, 64)
        self.down_block1 = DownBlock(64, 128)
        self.down_block2 = DownBlock(128, 256)
        self.down_block3 = DownBlock(256, 512)
        self.bottleneck = ResidualBlock(512)
        self.up_block1 = UpBlock(512, 256)
        self.up_block2 = UpBlock(256, 128)
        self.up_block3 = UpBlock(128, 64)
        self.one_one = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=3,
                      kernel_size=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # print(x.shape)
        e0 = self.input_block(x)
        # print(e0.shape)
        e1 = self.down_block1(e0)
        # print(e1.shape)
        e2 = self.down_block2(e1)
        # print(e2.shape)
        e3 = self.down_block3(e2)
        # print(e3.shape)
        b = self.bottleneck(e3)
        # print(b.shape)
        d1 = self.up_block1(e2, b)
        # print(d1.shape)
        d2 = self.up_block2(e1, d1)
        # print(d2.shape)
        d3 = self.up_block3(e0, d2)
        # print(d3.shape)
        y = self.one_one(d3)
        return y
        
    