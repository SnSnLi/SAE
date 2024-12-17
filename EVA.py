import torch
import torch.nn as nn
import math

def kernel_size(in_channel):  # Calculate the kernel size for 1D convolution, using ECA attention parameters [dynamic convolution kernel]
    k = int((math.log2(in_channel) + 1) // 2)
    return k + 1 if k % 2 == 0 else k


class MultiScaleFeatureExtractor(nn.Module):  # Multi-scale feature extractor [to strengthen the extraction of features from T1 and T2 at different time points with various sizes of convolution kernels]

    def __init__(self, in_channel, out_channel):
        super().__init__()
        # Use different sizes of convolution kernels for feature extraction
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=7, padding=3)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Perform convolution operations with different sizes of kernels
        out1 = self.relu(self.conv1(x))
        out2 = self.relu(self.conv2(x))
        out3 = self.relu(self.conv3(x))
        out = out1 + out2 + out3  # Sum up the features from different scales
        return out


class ChannelAttention(nn.Module):

    def __init__(self, in_channel):
        super().__init__()
        # Use adaptive average pooling and max pooling to extract global information
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.k = kernel_size(in_channel)
        # Use 1D convolution to calculate channel attention
        self.channel_conv1 = nn.Conv1d(4, 1, kernel_size=self.k, padding=self.k // 2)
        self.channel_conv2 = nn.Conv1d(4, 1, kernel_size=self.k, padding=self.k // 2)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, t1, t2):
        # Perform average and max pooling on t1 and t2
        t1_channel_avg_pool = self.avg_pool(t1)
        t1_channel_max_pool = self.max_pool(t1)
        t2_channel_avg_pool = self.avg_pool(t2)
        t2_channel_max_pool = self.max_pool(t2)
        # Concatenate the pooling results and transpose dimensions
        channel_pool = torch.cat([
            t1_channel_avg_pool, t1_channel_max_pool,
            t2_channel_avg_pool, t2_channel_max_pool
        ], dim=2).squeeze(-1).transpose(1, 2)
        # Use 1D convolution to calculate channel attention
        t1_channel_attention = self.channel_conv1(channel_pool)
        t2_channel_attention = self.channel_conv2(channel_pool)
        # Stack and normalize using Softmax
        channel_stack = torch.stack([t1_channel_attention, t2_channel_attention], dim=0)
        channel_stack = self.softmax(channel_stack).transpose(-1, -2).unsqueeze(-1)
        return channel_stack


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # Use 2D convolution to calculate spatial attention
        self.spatial_conv1 = nn.Conv2d(4, 1, kernel_size=7, padding=3)
        self.spatial_conv2 = nn.Conv2d(4, 1, kernel_size=7, padding=3)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, t1, t2):
        # Calculate the mean and max values of t1 and t2
        t1_spatial_avg_pool = torch.mean(t1, dim=1, keepdim=True)
        t1_spatial_max_pool = torch.max(t1, dim=1, keepdim=True)[0]
        t2_spatial_avg_pool = torch.mean(t2, dim=1, keepdim=True)
        t2_spatial_max_pool = torch.max(t2, dim=1, keepdim=True)[0]
        # Concatenate the mean and max values
        spatial_pool = torch.cat([
            t1_spatial_avg_pool, t1_spatial_max_pool,
            t2_spatial_avg_pool, t2_spatial_max_pool
        ], dim=1)
        # Use 2D convolution to calculate spatial attention
        t1_spatial_attention = self.spatial_conv1(spatial_pool)
        t2_spatial_attention = self.spatial_conv2(spatial_pool)
        # Stack and normalize using Softmax
        spatial_stack = torch.stack([t1_spatial_attention, t2_spatial_attention], dim=0)
        spatial_stack = self.softmax(spatial_stack)
        return spatial_stack


class TFAM(nn.Module):
    """Temporal Fusion Attention Module"""

    def __init__(self, in_channel):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channel)
        self.spatial_attention = SpatialAttention()

    def forward(self, t1, t2):
        # Calculate channel and spatial attention
        channel_stack = self.channel_attention(t1, t2)
        spatial_stack = self.spatial_attention(t1, t2)
        # Weighted sum and fusion
        stack_attention = channel_stack + spatial_stack + 1
        fuse = stack_attention[0] * t1 + stack_attention[1] * t2
        return fuse


class BFM(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.multi_scale_extractor = MultiScaleFeatureExtractor(in_channel, in_channel)
        self.tfam = TFAM(in_channel)

    def forward(self, t1, t2):
        # Perform multi-scale feature extraction
        t1_multi_scale = self.multi_scale_extractor(t1)
        t2_multi_scale = self.multi_scale_extractor(t2)
        # Fuse using TFAM
        output = self.tfam(t1_multi_scale, t2_multi_scale)
        return output


class EVA(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(EVA, self).__init__()
        self.bfm = BFM(in_channel)
        self.lstm = nn.LSTM(input_size=in_channel, hidden_size=128, num_layers=2, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, t1, t2):
        # Use BFM for feature fusion
        fused_features = self.bfm(t1, t2)

        # LSTM processes the fused feature sequence
        lstm_out, _ = self.lstm(fused_features.unsqueeze(0))  # Input is (batch, seq_len, features)
        lstm_out = lstm_out.squeeze(0)  # Remove batch dimension

        # Fully connected layer for classification
        output = self.fc(lstm_out)
        return output