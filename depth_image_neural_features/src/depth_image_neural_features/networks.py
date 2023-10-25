import torch
import torch.nn as nn
from depth_image_dataset.dataset import DepthImageDistanceFeaturesDataset
from torchsummary import summary
from torch.utils.data import DataLoader

class ConvDistEstimator(nn.Module):
    def __init__(self):
        super(ConvDistEstimator, self).__init__()
        self.cn1 = nn.Conv2d(1, 4, (3, 3),padding=(0,1), padding_mode="circular")
        self.cn2 = nn.Conv2d(4, 8, (3, 3),padding=(0,1), padding_mode="circular")
        self.cn3 = nn.Conv2d(8, 16, (3, 3),padding=(0,1), padding_mode="circular")
        self.cn4 = nn.Conv2d(16, 32, (3, 3),padding=(0,1), padding_mode="circular")
        self.cn5 = nn.Conv2d(32, 64, (3, 3),padding=(0,1), padding_mode="circular")
        self.cn6 = nn.Conv2d(64, 128, (3, 3),padding=(0,1), padding_mode="circular")
        self.fcc1 = nn.Linear(128,64)
        self.fcc2 = nn.Linear(64,32)
        self.fcd1 = nn.Linear(64,32)
        self.fcd2 = nn.Linear(32,16)
        self.fcd3 = nn.Linear(16,8)
        self.fcd4 = nn.Linear(8,1)
        self.mpool = nn.MaxPool2d((1, 2))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()

    def compress(self, x):
        x = self.mpool(self.relu(self.cn1(x)))
        x = self.mpool(self.relu(self.cn2(x)))
        x = self.mpool(self.relu(self.cn3(x)))
        x = self.mpool(self.relu(self.cn4(x)))
        x = self.mpool(self.relu(self.cn5(x)))
        x = self.mpool(self.relu(self.cn6(x)))
        x = torch.reshape(self.avgpool(x),(-1,128))
        x = self.relu(self.fcc1(x))
        x = self.relu(self.fcc2(x))
        return x

    def distance(self, x,y):
        z = torch.cat((x,y),-1)
        z = self.relu(self.fcd1(z))
        z = self.relu(self.fcd2(z))
        z = self.relu(self.fcd3(z))
        z = self.fcd4(z)
        return z

    def forward(self, images):
        x,y = images
        x = self.compress(x)
        y = self.compress(y)
        z =self.distance(x,y)
        return z



