from depth_image_neural_features.networks import DeepDistEstimator
from depth_image_dataset.dataset import DepthImageDistanceFeaturesDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
torch.set_float32_matmul_precision('high')


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
for file in os.listdir(logdir):
    os.remove(os.path.join(logdir, file))
writer = SummaryWriter(logdir)
# Load dataset:
dataset = DepthImageDistanceFeaturesDataset(
    name="depth_image_with_3_stackings", samples_to_generate=1000000
)
criterion1 = nn.MSELoss(3)
criterion2 = nn.MSELoss(4)
# Training loop
bs = 64
n_epochs = 64
alfa = 0.3
beta = 0.5
for lr in tqdm([0.0001], desc="Lr"):
    model = DeepDistEstimator().to("cuda")
    model =torch.compile(model)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    counter = 0
    for n_epoch in tqdm(range(n_epochs), desc="Epoch", leave=False):
        dataloader = DataLoader(dataset, bs, shuffle=True)
        model.to("cuda")
        for sample in tqdm(dataloader, leave=False):
            # Get data
            (img1, img2), labels = sample
            img1 = img1.to("cuda")
            img2 = img2.to("cuda")
            labels = labels.to("cuda")
            # Forward pass
            optimizer.zero_grad()
            outputs, aux_outputs1, aux_outputs2 = model(img1, img2)
            # calculate loss
            main_pose_loss, main_orientation_loss = calculate_loss(outputs,         labels, criterion1, criterion2)
            aux1_pose_loss, aux1_orientation_loss = calculate_loss(aux_outputs1,    labels, criterion1, criterion2)
            aux2_pose_loss, aux2_orientation_loss = calculate_loss(aux_outputs2,    labels, criterion1, criterion2)
            main_loss = main_pose_loss + beta*main_orientation_loss
            aux1_loss = aux1_pose_loss + beta*aux1_orientation_loss
            aux2_loss = aux2_pose_loss + beta*aux2_orientation_loss
            total_loss = main_loss + (aux1_loss  + aux2_loss) * alfa
            total_loss.backward()
            optimizer.step()
            writer.add_scalar(f"{lr}/total_loss", total_loss, counter)
            writer.add_scalar(f"{lr}/main_loss/total", main_loss, counter)
            writer.add_scalar(f"{lr}/main_loss/pose", main_pose_loss, counter)
            writer.add_scalar(f"{lr}/main_loss/orientation", main_orientation_loss, counter)
            writer.add_scalar(f"{lr}/aux_loss_1/total", aux1_loss, counter)
            writer.add_scalar(f"{lr}/aux_loss_1/pose", aux1_pose_loss, counter)
            writer.add_scalar(f"{lr}/aux_loss_1/orientation", aux1_orientation_loss, counter)
            writer.add_scalar(f"{lr}/aux_loss_2/total", aux2_loss, counter)
            writer.add_scalar(f"{lr}/aux_loss_2/pose", aux2_pose_loss, counter)
            writer.add_scalar(f"{lr}/aux_loss_2/orientation", aux2_orientation_loss, counter)
            counter += 1
        torch.save(
            model.to("cpu").state_dict(),
            f"/home/lorenzo/models/{model.__class__.__name__}_{lr}_{bs}_{alfa}_{beta}.torch",
        )
