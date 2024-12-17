import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models

def global_median_pooling(x):
    median_pooled = torch.median(x.view(x.size(0), x.size(1), -1), dim=2)[0]
    median_pooled = median_pooled.view(x.size(0), x.size(1), 1, 1)
    return median_pooled

class ChannelAttention(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(input_channels, internal_neurons, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(internal_neurons, input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        avg_pool = F.adaptive_avg_pool2d(inputs, (1, 1))
        max_pool = F.adaptive_max_pool2d(inputs, (1, 1))
        median_pool = global_median_pooling(inputs)

        avg_out = self.fc2(F.relu(self.fc1(avg_pool), inplace=True))
        max_out = self.fc2(F.relu(self.fc1(max_pool), inplace=True))
        median_out = self.fc2(F.relu(self.fc1(median_pool), inplace=True))

        out = torch.sigmoid(avg_out + max_out + median_out)
        return out

class MECS(nn.Module):
    def __init__(self, in_channels, out_channels, channel_attention_reduce=4):
        super(MECS, self).__init__()
        self.C = in_channels
        self.O = out_channels
        assert in_channels == out_channels

        self.channel_attention = ChannelAttention(
            input_channels=in_channels,
            internal_neurons=in_channels // channel_attention_reduce
        )

        self.initial_depth_conv = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=5, padding=2, 
            groups=in_channels
        )

        self.depth_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 7), padding=(0, 3), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(7, 1), padding=(3, 0), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 11), padding=(0, 5), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(11, 1), padding=(5, 0), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 21), padding=(0, 10), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(21, 1), padding=(10, 0), groups=in_channels),
        ])

        self.pointwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, inputs):
        inputs = self.pointwise_conv(inputs)
        inputs = self.act(inputs)

        channel_att = self.channel_attention(inputs)
        inputs = channel_att * inputs

        initial_out = self.initial_depth_conv(inputs)
        spatial_outs = [conv(initial_out) for conv in self.depth_convs]
        spatial_out = sum(spatial_outs)

        spatial_att = self.pointwise_conv(spatial_out)
        out = spatial_att * inputs
        out = self.pointwise_conv(out)
        return out

class ResNet50WithMECS(models.ResNet):
    def __init__(self, pretrained=True, num_classes=1000):
        super(ResNet50WithMECS, self).__init__(
            block=models.resnet.Bottleneck,
            layers=[3, 4, 6, 3],
            num_classes=num_classes
        )
        
        if pretrained:
            state_dict = models.resnet50(pretrained=True).state_dict()
            self.load_state_dict(state_dict, strict=False)
            
        in_channels = self.layer4[-1].conv3.out_channels
        self.mecs = MECS(in_channels, in_channels)
        
        if num_classes != 1000:
            self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.mecs(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

def create_model(num_classes=7, pretrained=True):
    model = ResNet50WithMECS(pretrained=pretrained, num_classes=num_classes)
    return model
