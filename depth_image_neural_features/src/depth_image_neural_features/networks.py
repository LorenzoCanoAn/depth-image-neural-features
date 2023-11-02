import torch
import torch.nn as nn
from depth_image_dataset.dataset import DepthImageDistanceFeaturesDataset
from torchsummary import summary
from torch.utils.data import DataLoader
from depth_image_neural_features.googlenet import GoogLeNetCompressor


class FeatureDistanceEstimator(nn.Module):
    def __init__(self, input_size, dropout):
        super(FeatureDistanceEstimator, self).__init__()
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


class DeepDistEstimator(nn.Module):
    def __init__(self,feature_vector_size=128):
        super(DeepDistEstimator, self).__init__()
        self.compressor = GoogLeNetCompressor(feature_vector_size)
        self.distance = FeatureDistanceEstimator(int(feature_vector_size *2 ),0.5)

    def forward(self, x, y):
        x, xaux1, xaux2 = self.compressor(x)
        y, yaux1, yaux2= self.compressor(y)
        z = self.distance(x, y)
        zaux1, zaux2 = None, None
        if self.training:
            zaux1 = self.distance(xaux1, yaux1)
            zaux2 = self.distance(xaux2, yaux2)
        return z, zaux1, zaux2


if __name__ == "__main__":
    img = torch.ones((10, 1, 128, 128)).to("cuda")
    model =DeepDistEstimator().to("cuda")
    summary(model, input_data=(img, img))
    model.train()
    zs =model(img,img)
    for z in zs:
        print(z) 
    
