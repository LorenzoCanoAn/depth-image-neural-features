from depth_image_neural_features.networks import *
from depth_image_dataset.dataset import DepthImageDistanceFeaturesDataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import os

torch.set_float32_matmul_precision("high")


def calculate_loss(outputs, labels, criterion1, criterion2):
    label_poses = labels[:, :3]
    label_orentations = labels[:, 3:]
    estimated_poses = outputs[:, :3]
    estimated_orientations = outputs[:, 3:]
    pose_loss = criterion1(estimated_poses, label_poses)
    orientation_loss = criterion2(estimated_orientations, label_orentations)
    return pose_loss, orientation_loss


# Setup loging
logdir = "/home/lorenzo/.tensorboard"
os.popen(f"rm -rf {logdir}/**")
writer = SummaryWriter(logdir)
# MODEL
model = LastHope()
#model.load_state_dict(torch.load("/home/lorenzo/models/LastHope_lr0.004_bs128_nepochs64_fvsize256_ptchsize8/57.torch"),strict=False)
# Load dataset:
dataset = DepthImageDistanceFeaturesDataset(
    name="large_dataset", samples_to_generate=50,samples_to_load=800000
)
criterion1 = nn.MSELoss(2,reduction="mean")
# Training loop
n_epochs = 128
bs = 128
image_size = 128
patch_size = 8
num_layers = 8
num_heads = 8
hidden_dims = 32
mlp_dim = 256
dropout = 0.1
attention_dropout = 0.1
feature_length = 256
lr = 0.004
for n_samples in [2000000]:
    dataset.samples_to_generate = n_samples
    dataset.process_raw_inputs()
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.9,0.1])
    train_dataloader = DataLoader(train_dataset, bs, shuffle=True)
    save_dir = f"/home/lorenzo/models/retrain_last_hope_with_dropout"
    os.makedirs(save_dir,exist_ok=True)
    model.train()
    counter = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = ExponentialLR(optimizer,0.95)
    for n_epoch in tqdm(range(n_epochs), desc="Epoch", leave=True):
        model.to("cuda")
        for sample in tqdm(train_dataloader, leave=True):
            model.train()
            # Get data
            (img1, img2), labels = sample
            img1 = img1.to("cuda")
            img2 = img2.to("cuda")
            noise1 = torch.tensor(np.random.normal(size=img1.shape)*2/100 +1).to("cuda")
            noise2 = torch.tensor(np.random.normal(size=img2.shape)*2/100 +1).to("cuda")
            img1 *= noise1
            img2 *= noise2
            labels = labels.to("cuda")
            # Forward pass
            optimizer.zero_grad()
            result = model(img1, img2)
            # calculate loss
            train_loss = criterion1(result, labels)
            train_loss.backward()
            optimizer.step()
            writer.add_scalars(f"{n_samples}/results",{"lx":labels[0,0],"rx":result[0,0],"ly":labels[0,1],"ry":result[0,1]}, counter)
            counter += 1
            # test data
            (img1, img2), labels = test_dataset[np.random.randint(0,len(test_dataset))]
            img1 = img1.to("cuda")
            img2 = img2.to("cuda")
                # Add batch dimension
            img1 = torch.unsqueeze(img1,0) 
            img2 = torch.unsqueeze(img2,0)
            labels = torch.unsqueeze(labels,0)
            labels = labels.to("cuda")
            model = model.eval()
            result = model(img1, img2)
            test_loss = criterion1(result, labels)
            writer.add_scalars(f"{n_samples}/loss",{"train_loss":train_loss,"test_loss":test_loss}, counter)
        lr_scheduler.step()
        torch.save(
            model.to("cpu").state_dict(),
            os.path.join(save_dir,f"{n_epoch}.torch")
        )
