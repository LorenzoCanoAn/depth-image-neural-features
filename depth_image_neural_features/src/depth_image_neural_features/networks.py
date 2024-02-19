import torch
import torch.nn as nn
from torchinfo import summary
from torch.utils.data import DataLoader
from depth_image_neural_features.googlenet import GoogLeNetCompressor
from depth_image_neural_features.resnet import ResNet
from depth_image_neural_features.transformer import (
    TransformerEncoder,
    TransformerEncoderLayer,
)


class FcDistanceEstimator(nn.Module):
    def __init__(self, input_size, dropout):
        super(FcDistanceEstimator, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 7)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y) -> torch.Tensor:
        z = torch.cat((torch.flatten(x, 1), torch.flatten(y, 1)), -1)
        z = self.dropout(self.relu(self.fc1(z)))
        z = self.dropout(self.relu(self.fc2(z)))
        z = self.dropout(self.relu(self.fc3(z)))
        z = self.dropout(self.fc4(z))
        return z


class TrnsfDistanceEstimator(nn.Module):
    def __init__(self, n_features):
        self.n_features = n_features

    def forward(self, x, y):
        assert x.shape[-1] == self.n_features
        pass


class GNetFcDistEstimator(nn.Module):
    def __init__(self, feature_vector_size=128):
        super(GNetFcDistEstimator, self).__init__()
        self.compressor = GoogLeNetCompressor(feature_vector_size)
        self.distance = FcDistanceEstimator(int(feature_vector_size * 2), 0.5)

    def forward(self, x, y):
        x, xaux1, xaux2 = self.compressor(x)
        y, yaux1, yaux2 = self.compressor(y)
        z = self.distance(x, y)
        zaux1, zaux2 = None, None
        if self.training:
            zaux1 = self.distance(xaux1, yaux1)
            zaux2 = self.distance(xaux2, yaux2)
        return z, zaux1, zaux2


class ResnetAndTransformer(nn.Module):
    def __init__(self):
        super(type(self), self).__init__()
        self.resnet = ResNet()
        self.transformer = TransformerEncoder(TransformerEncoderLayer(32, 8, 512), 8)
        self.fc1 = nn.Linear(32, 1)
        self.fc2 = nn.Linear(64, 3)
        self.relu = nn.ReLU()

    def encode(self, x):
        x = self.resnet(x)  # Nx32x1x32
        x = torch.squeeze(x)  # Nx32x32
        x.permute((0, 2, 1))
        return x

    def dif(self, x, y):
        z = torch.cat([x, y], -2)
        z = self.transformer(z)
        z = self.relu(self.fc1(z))
        z = torch.squeeze(z)
        z = self.relu(self.fc2(z))
        return z

    def forward(self, x, y):
        x = self.encode(x)
        y = self.encode(y)
        z = self.dif(x, y)
        return z


class LastHope(nn.Module):
    def __init__(self) -> None:
        super(type(self), self).__init__()
        fs = int(124*2*2)
        self.compressor = nn.Sequential(
            nn.Conv2d(1, 16, 5, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 16, (3,5), (1, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 8, (3,5), (1, 2)),
            nn.ReLU(),
            nn.Conv2d(8, 2, (3,5), (1, 2)),
        )
        self.compressor2 = nn.Sequential(
            nn.Linear(fs, fs),
            nn.ReLU(),
            nn.Linear(fs, fs),
            nn.ReLU(),
            nn.Linear(fs, fs),
            nn.ReLU(),
            nn.Linear(fs, fs),
        )
        self.feature_comparator = nn.Sequential(
            nn.Linear(fs, fs),
            nn.ReLU(),
            nn.Linear(fs, fs),
            nn.ReLU(),
            nn.Linear(fs, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def compress(self, img) -> torch.Tensor:
        z = self.compressor(img)
        assert isinstance(z, torch.Tensor)
        z = z.flatten(1)
        z = self.compressor2(z)
        return z

    def compare_features(self, z1, z2):
        return self.feature_comparator(z2 - z1)

    def forward(self, img1, img2):
        z1 = self.compress(img1)
        z2 = self.compress(img2)
        if self.training:
            z1 = z1 + torch.normal(0,0.5,size=z1.shape).to("cuda")
            z2 = z2 + torch.normal(0,0.5,size=z2.shape).to("cuda")
        return self.compare_features(z1, z2)
