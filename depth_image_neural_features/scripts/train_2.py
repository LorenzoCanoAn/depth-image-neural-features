from depth_image_neural_features.networks import *
from torch.utils.data import DataLoader
import torch
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import os
import json

torch.set_float32_matmul_precision("high")


def random_non_repeat_ints(max_int, num_ints) -> np.ndarray:
    n_repeats = int(np.floor(num_ints / max_int))
    if n_repeats == 0:
        ints = np.arange(0, max_int, dtype=int)
        np.random.shuffle(ints)
        ints = ints[:num_ints]
    else:
        ints = np.arange(0, max_int, dtype=int)
        np.random.shuffle(ints)
        ints = ints[: (num_ints - n_repeats * max_int)]
        for _ in range(n_repeats):
            ints_to_cat = np.arange(0, max_int, dtype=int)
            np.random.shuffle(ints_to_cat)
            ints = np.concatenate((ints, ints_to_cat))
    return ints

class IndexBasedDataset:
    def __init__(self, path_to_index, max_distance):
        self.max_distance = max_distance
        with open(path_to_index, "r") as f:
            print("Loading index")
            self.index = json.load(f)
            print("Index loaded")
        self.data = self.index["data"]
        self.n_images_per_world = dict()
        for key in self.data.keys():
            self.n_images_per_world[key] = len(self.data[key]["poses"])
        self.n_images = sum(
            [self.n_images_per_world[key] for key in self.n_images_per_world.keys()]
        )

    def sum_dict(self, d):
        suma = 0
        for key in d.keys():
            suma += d[key]
        return suma

    def generate_n_pairs_per_world(self, n_pairs):
        pairs_per_world = dict()
        for world_name in self.data.keys():
            pairs_per_world[world_name] = int(
                n_pairs / self.n_images * self.n_images_per_world[world_name]
            )
        n_iter = 0
        keys = list(pairs_per_world.keys())
        while self.sum_dict(pairs_per_world) < n_pairs:
            n_key = n_iter % len(keys)
            key = keys[n_key]
            pairs_per_world[key] += 1
        return pairs_per_world

    def generate_pairs(self, n_pairs):
        n_pairs_per_world = self.generate_n_pairs_per_world(n_pairs)
        self.n_pairs = n_pairs
        self.pairs_index = [["world", 0, 0, np.zeros((0,0))] for _ in range(n_pairs)]
        n_pair = 0
        for world_name in tqdm(self.data.keys(),total=len(list(self.data.keys()))):
            n_pairs_for_this_world = n_pairs_per_world[world_name]
            points_of_this_world = self.data[world_name]["poses"]
            idxs1 = random_non_repeat_ints(
                len(points_of_this_world), n_pairs_for_this_world
            )
            for idx1 in tqdm(idxs1,total=len(idxs1),leave=False,desc=world_name):
                while True:
                    idx2 = np.random.randint(0,len(points_of_this_world))
                    p1 = np.array(points_of_this_world[idx1][0:3])
                    p2 = np.array(points_of_this_world[idx2][0:3])
                    vector = p2 - p1
                    d = np.linalg.norm(vector,2)
                    if d < self.max_distance:
                        self.pairs_index[n_pair][0] = world_name
                        self.pairs_index[n_pair][1] =  idx1
                        self.pairs_index[n_pair][2] =  idx2
                        self.pairs_index[n_pair][3] =  torch.tensor(vector[:2].astype(np.float32))
                        n_pair += 1
                        break
        self.load_images()
    def load_images(self):
        loading_set = set()
        for index_element in tqdm(self.pairs_index,desc="gen loading set"):
            wn, id1, id2, v = index_element
            loading_set.add((wn, id1))
            loading_set.add((wn, id2))
        loading_set = list(loading_set)
        print("Alocating_memory")
        self.images = torch.zeros((len(loading_set),1,16,1024))
        self.images_dict = dict()
        for n, (wn, idx) in enumerate(loading_set):
            self.images_dict[(wn, idx)] = n
        for n, (wn, idx) in tqdm(enumerate(loading_set),desc="Loading images", total=len(loading_set)):
            self.images[n,0] = torch.Tensor(np.load(os.path.join(self.data[wn]["images_folder_path"],f"{idx:010d}.npy")))
    
    def __len__(self):
        return self.n_pairs

    def __getitem__(self, idx):
        wn, id1, id2, v = self.pairs_index[idx]
        img1 = self.images[self.images_dict[(wn, id1)]]
        img2 = self.images[self.images_dict[(wn, id2)]]
        return (img1, img2), v
n_samples = 500000
dataset = IndexBasedDataset("/home/lorenzo/.datasets/comprehensive_depth_image_dataset/index.json", max_distance=3)
dataset.generate_pairs(n_samples)
# Setup loging
logdir = "/home/lorenzo/.tensorboard"
os.popen(f"rm -rf {logdir}/**")
writer = SummaryWriter(logdir)
# MODEL
model = LastHope()
criterion1 = nn.MSELoss(2, reduction="mean")
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
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
train_dataloader = DataLoader(train_dataset, bs, shuffle=True)
save_dir = f"/home/lorenzo/models/retrain_last_hope_with_dropout"
os.makedirs(save_dir, exist_ok=True)
model.train()
counter = 0
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
lr_scheduler = ExponentialLR(optimizer, 0.95)
for n_epoch in tqdm(range(n_epochs), desc="Epoch", leave=True):
    model.to("cuda")
    for sample in tqdm(train_dataloader, leave=True):
        model.train()
        # Get data
        (img1, img2), labels = sample
        img1 = img1.to("cuda")
        img2 = img2.to("cuda")
        noise1 = torch.tensor(np.random.normal(size=img1.shape) * 2 / 100 + 1).to(
            "cuda"
        )
        noise2 = torch.tensor(np.random.normal(size=img2.shape) * 2 / 100 + 1).to(
            "cuda"
        )
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
        writer.add_scalars(
            f"{n_samples}/results",
            {
                "lx": labels[0, 0],
                "rx": result[0, 0],
                "ly": labels[0, 1],
                "ry": result[0, 1],
            },
            counter,
        )
        counter += 1
        # test data
        (img1, img2), labels = test_dataset[np.random.randint(0, len(test_dataset))]
        img1 = img1.to("cuda")
        img2 = img2.to("cuda")
        # Add batch dimension
        img1 = torch.unsqueeze(img1, 0)
        img2 = torch.unsqueeze(img2, 0)
        labels = torch.unsqueeze(labels, 0)
        labels = labels.to("cuda")
        model = model.eval()
        result = model(img1, img2)
        test_loss = criterion1(result, labels)
        writer.add_scalars(
            f"{n_samples}/loss",
            {"train_loss": train_loss, "test_loss": test_loss},
            counter,
        )
    lr_scheduler.step()
    torch.save(
        model.to("cpu").state_dict(), os.path.join(save_dir, f"{n_epoch}.torch")
    )
