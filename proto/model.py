import torch
import torch.nn as nn
from torchvision import models

def conv_block(in_channels, out_channels):
    batch_norm = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(batch_norm.weight)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        batch_norm,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class ProtoNet(nn.Module):
    def __init__(self, x_dim=1, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim)
        )

#         self.linear1 = nn.Linear(576, 64)
#         self.linear2 = nn.Linear(64, 32)
#         self.relu = nn.ReLU()
        
        
    def forward(self, x):
        out = self.encoder(x)
        out = out.view(out.size()[0], -1)
#         out = self.relu(self.linear1(out))
#         out = self.linear2(out)

        return out
