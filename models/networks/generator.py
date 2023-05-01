import torch.nn as nn
import torch.nn.functional as F
import torch
from models.networks.base_network import BaseNetwork
from torchvision.models import vgg19

# Weights initializer
def weights_init_normal(m):
    """
    Initilize weights with normal distribution to the networks
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# Normalization block
class Norm_layer(nn.Module):
    def __init__(self,ch):
        super(Norm_layer, self).__init__()
        self.norm = nn.BatchNorm2d(ch, affine=True)

    def forward(self, x):
        Norm = self.norm(x)
        return Norm 
    

# Residual block
class ResBl(nn.Module):
    """
    This is the residual block class, this block is part of the main block (MainBl) 
    Args:
        in_size: number of input channels
        out_size: number of input channels
    """
    def __init__(self, in_size, out_size):
        super(ResBl, self).__init__()

        self.conv1 = (nn.Conv2d(in_size, (in_size + out_size)//2, 3, 1, 1, bias=True))
        self.SA1 = Norm_layer((in_size))
        
        self.conv2 = (nn.Conv2d((in_size + out_size)//2, (in_size + out_size)//2, 3, 1, 1, bias=True))
        self.SA2 = Norm_layer((in_size + out_size)//2)
        
        self.conv3 =(nn.Conv2d((in_size + out_size)//2, out_size, 3, 1, 1, bias=True))
        self.SA3 = Norm_layer((in_size + out_size)//2)

    def forward(self, x):
        out1 = F.relu(self.SA1((x)))
        out2 = self.conv1(out1)
        out3 = F.relu(self.SA2((out2)))
        out4 = self.conv2(out3)
        out5 =  F.relu(self.SA3((out4)))
        out6 = self.conv3(out5)

        return out6


# Concatenation block
class UpFeat(nn.Module):
    """
    This is the Concatenation block class, this block is part of the main block (MainBl) 
    and concatenate the feature maps from previous up-sampling block and corresponding down-sampling block
    NL represents Normalization Layer
    out_n represents the output after n number of operations
    Args:
        in_size: number of input channels
        out_size: number of input channels
    """
    def __init__(self, in_size, out_size):
        super(UpFeat, self).__init__()

        self.conv1 = (nn.Conv2d(in_size, out_size, 3, 1, 1, bias=True))
        self.SA1 = Norm_layer(out_size)
        self.conv2 = (nn.Conv2d(out_size, out_size, 3, 1, 1, bias=True))
    def forward(self, x):
        
        
        out1 = self.conv1(x)
        out2 = F.relu(self.SA1((out1)))
        out3 = self.conv2(out2)
        
        return out3


class MainBl(nn.Module):
    """ 
    The implementation of the main block class which is used in down-sampling block, bridge block, and up-sampling block
    NL represents Normaization Layer
    out_n represents the output after n number of operations
    resbl is an instantiation of residual block
    Args:
        in_size: number of input channels
        out_size: number of input channels
        
    """

    def __init__(self, in_size, out_size):
        super(MainBl, self).__init__()

        self.resbl = ResBl(in_size, out_size)

        self.conv = nn.Conv2d(in_size, out_size, 3, 1, 1, bias=True)
        self.SA = Norm_layer(in_size)

    def forward(self, x):
        out1 = self.resbl(x)
        out2 =  F.relu(self.SA((x)))
        out3 = self.conv(out2)
        return out1 + out3


class GeneratorUNet(BaseNetwork):

    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.pooling = nn.MaxPool2d(2, stride=2)
        self.pooling2 = nn.AvgPool2d(2, stride=2)
        self.upsampling = nn.Upsample(scale_factor=2, mode="bilinear")

        self.ch1 = nn.Sequential(nn.Conv2d(in_channels, 32, 3, 1, 1, bias=True))
        self.ch2 = nn.Sequential(nn.Conv2d(in_channels, 64, 3, 1, 1, bias=True))
        self.ch3 = nn.Sequential((nn.Conv2d(in_channels, 128, 3, 1, 1, bias=True)))
        self.ch4 = nn.Sequential((nn.Conv2d(in_channels, 256, 3, 1, 1, bias=True)))

        self.ch5 = UpFeat(512,256)
        self.ch6 = UpFeat(256,128)
        self.ch7 = UpFeat(128,64)
        
        self.down1 = MainBl(32, 64)
        self.down2 = MainBl(64, 128)
        self.down3 = MainBl(128, 256)
        self.down4 = MainBl(256, 256)
        
        self.up1 = MainBl(256, 128)
        self.up2 = MainBl(128, 64)
        self.up3 = MainBl(64, 32)

        self.conv = nn.Sequential(nn.Conv2d(32, 3, 3, 1, 1, bias=True))
        # For segmentation task, we remove the Tanh activation function
    def forward(self, x):

        # The implementation of the UResNet consisting of three down-sampling block, a bridge block, and three up-sampling block
        # This network serves as the generator of our framwork
        # chN and conN is for matching the number of feature maps with the corresponding block
        # dN represents feature maps resulting from a block before max poooling or up-sampling
        # dN_ represents feature maps resulting from a block after max poooling or up-sampling
        # x is the input and c is the conditions
        
        d1 = self.ch1(x)  
        d2 = self.down1((d1))
        d2_ = self.pooling(d2)  
        x_1 = self.pooling2(x)  
        
        d3 = self.ch2(x_1)
        d4 = self.down2((d3+d2_))
        d4_ = self.pooling(d4)  
        x_2 = self.pooling2(x_1) 
    
        d5 = self.ch3(x_2)
        d6 = self.down3((d5+d4_))
        d6_ = self.pooling(d6)  
        x_3 = self.pooling2(x_2) 
        
        d7 = self.ch4(x_3)
        d8 = self.down4((d7+d6_))
        u1 = self.upsampling(d8)
        M1 = torch.cat((d6, u1), dim=1)
     
        d9 = self.ch5((M1))
        d10 = self.up1((d9))
        u2 = self.upsampling(d10)
        M2 = torch.cat((d4, u2), dim=1)
        
        d11 = self.ch6((M2))
        d12 = self.up2((d11))
        u3 = self.upsampling(d12)
        M3 = torch.cat((d2, u3), dim=1)      
        
        d13 = self.ch7((M3))
        d14 = self.up3((d13))
        
        uf = self.conv(d14)

        return (uf)
    
    
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out