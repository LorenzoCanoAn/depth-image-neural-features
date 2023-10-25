from depth_image_neural_features.networks import ConvDistEstimator
from depth_image_dataset.dataset import DepthImageDistanceFeaturesDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# Load dataset:
dataset = DepthImageDistanceFeaturesDataset(
    name="test_dataset_depth_image", samples_to_generate=100000
)

writer = SummaryWriter("/home/lorenzo/.tensorboard")
criterion = nn.MSELoss(1)
# Training loop
for lr in tqdm([0.005,0.001, 0.0005,0.0001, 0.00005, 0.00001, 0.000005, 0.000001], desc="Lr"):
    model = ConvDistEstimator().to("cuda")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    counter = 0
    for n_epoch in tqdm(range(8), desc="Epoch"):
        if n_epoch != 0:
            dataset.process_raw_inputs()
        dataloader = DataLoader(dataset,16,shuffle=True)
        for sample in tqdm(dataloader,leave=False):
            optimizer.zero_grad()
            (img1, img2), labels = sample
            img1 = img1.to("cuda")
            img2 = img2.to("cuda")
            labels = labels.to("cuda")
            outputs = model((img1, img2))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            writer.add_scalar(f"loss_per_iter_lr{lr}", loss, counter)
            counter += 1
    torch.save(model.to("cpu").state_dict(), f"/home/lorenzo/models/{model.__class__.__name__}_{lr}.torch")
        