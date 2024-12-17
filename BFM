import torch
import torch.nn as nn
import torch.nn.functional as F

def global_median_pool2d(x):
    return torch.median(x.view(x.size(0), x.size(1), -1), -1, keepdim=True)[0].unsqueeze(-1)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1)
        )
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        med_out = self.fc(global_median_pool2d(x))
        out = torch.sigmoid(avg_out + max_out + med_out)
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return torch.sigmoid(x)

class TFAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1)
        self.conv2 = nn.Conv2d(channels, channels, 1)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x1, x2):
        b,c,h,w = x1.shape
        query = self.conv1(x1).view(b,-1,h*w).permute(0,2,1)
        key = self.conv2(x2).view(b,-1,h*w)
        energy = torch.bmm(query, key)
        attention = self.softmax(energy)
        value = x2.view(b,-1,h*w)
        out = torch.bmm(value, attention.permute(0,2,1))
        out = out.view(b,c,h,w)
        return out

class BFM(nn.Module):
    def __init__(self, in_channels, hidden_size=256):
        super().__init__()
        self.conv1x1_1 = nn.Conv2d(in_channels, in_channels, 1)
        self.conv1x1_2 = nn.Conv2d(in_channels, in_channels, 1) 
        self.gru = nn.GRU(in_channels, hidden_size, 2, bidirectional=True, batch_first=True)
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()
        self.tfam = TFAM(in_channels)
        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        
    def forward(self, x1, x2):
        b,c,h,w = x1.shape
        
        # Initial convs
        f1 = self.conv1x1_1(x1)
        f2 = self.conv1x1_2(x2)
        
        # Reshape and GRU
        f1_seq = f1.view(b,c,-1).permute(0,2,1)
        f2_seq = f2.view(b,c,-1).permute(0,2,1)
        
        out1, _ = self.gru(f1_seq)
        out2, _ = self.gru(f2_seq)
        
        # Reshape back
        out1 = out1.permute(0,2,1).view(b,c,h,w)
        out2 = out2.permute(0,2,1).view(b,c,h,w)
        
        # Channel attention
        ca1 = self.ca(out1)
        ca2 = self.ca(out2)
        out1 = ca1 * out1
        out2 = ca2 * out2
        
        # Spatial attention  
        sa1 = self.sa(out1)
        sa2 = self.sa(out2)
        out1 = sa1 * out1
        out2 = sa2 * out2
        
        # TFAM
        tfam_out = self.tfam(out1, out2)
        
        # Fusion
        out = self.conv_out(torch.cat([out1, tfam_out], dim=1))
        
        return out

if __name__ == '__main__':
    model = BFM(256)
    x1 = torch.randn(2, 256, 32, 32)
    x2 = torch.randn(2, 256, 32, 32)
    out = model(x1, x2)
    print(out.shape)
