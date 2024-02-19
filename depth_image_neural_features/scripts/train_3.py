from torch.utils.data import DataLoader
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import os
import json
import neptune


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


def stack_image(image, n_stackings):
    height, width = image.shape
    assert width % 2**n_stackings == 0
    for n_stack in range(1, n_stackings + 1):
        new_width = int(width / 2**n_stack)
        image = np.vstack((image[:, :new_width], image[:, new_width:]))
    return image


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

    def set_length(self, length):
        if length <= self.n_pairs:
            self._len = length
        else:
            raise Exception("The manually set length can't be longer that the number of pairs")

    def generate_pairs(self, n_pairs):
        self.n_pairs = n_pairs
        self._len = n_pairs
        n_pairs_per_world = self.generate_n_pairs_per_world(n_pairs)
        self.n_pairs = n_pairs
        self.pairs_index = [["world", 0, 0, np.zeros((0, 0))] for _ in range(n_pairs)]
        n_pair = 0
        for world_name in tqdm(self.data.keys(), total=len(list(self.data.keys()))):
            n_pairs_for_this_world = n_pairs_per_world[world_name]
            points_of_this_world = self.data[world_name]["poses"]
            idxs1 = random_non_repeat_ints(len(points_of_this_world), n_pairs_for_this_world)
            for idx1 in tqdm(idxs1, total=len(idxs1), leave=False, desc=world_name):
                while True:
                    idx2 = np.random.randint(0, len(points_of_this_world))
                    p1 = np.array(points_of_this_world[idx1][0:3])
                    p2 = np.array(points_of_this_world[idx2][0:3])
                    vector = p2 - p1
                    d = np.linalg.norm(vector, 2)
                    if d < self.max_distance:
                        self.pairs_index[n_pair][0] = world_name
                        self.pairs_index[n_pair][1] = idx1
                        self.pairs_index[n_pair][2] = idx2
                        self.pairs_index[n_pair][3] = torch.tensor(vector[:2].astype(np.float32))
                        n_pair += 1
                        break
        self.load_images()

    def load_images(self):
        loading_set = set()
        for index_element in tqdm(self.pairs_index, desc="gen loading set"):
            wn, id1, id2, v = index_element
            loading_set.add((wn, id1))
            loading_set.add((wn, id2))
        loading_set = list(loading_set)
        print("Alocating_memory")
        self.images = torch.zeros((len(loading_set), 1, 16, 1024))
        self.images_dict = dict()
        for n, (wn, idx) in enumerate(loading_set):
            self.images_dict[(wn, idx)] = n
        for n, (wn, idx) in tqdm(
            enumerate(loading_set), desc="Loading images", total=len(loading_set)
        ):
            self.images[n, 0] = torch.Tensor(
                np.load(os.path.join(self.data[wn]["images_folder_path"], f"{idx:010d}.npy"))
            )

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        wn, id1, id2, v = self.pairs_index[idx]
        img1 = self.images[self.images_dict[(wn, id1)]]
        img2 = self.images[self.images_dict[(wn, id2)]]
        return (img1, img2), v


dataset = IndexBasedDataset(
    "/home/lorenzo/datasets/comprehensive_depth_image_dataset/index.json", max_distance=3
)
n_samples = 500000
dataset.generate_pairs(n_samples)
# Setup loging

# MODEL
from depth_image_neural_features.networks import LastHope

run = neptune.init_run(
    project="lcano/depth-image-odom-features",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiYjcxZGU4OC00ZjVkLTRmMDAtYjBlMi0wYzkzNDQwOGJkNWUifQ==",
)
model = LastHope()
criterion1 = nn.MSELoss(2, reduction="mean")
# PARAMETERS
n_epochs = 1024
bs = 128
dropout = 0.1
lr = 0.001
save_dir = f"/home/lorenzo/models/retrain_last_hope_with_dropout"
dataset_increment = 5000
dataset_initial_length = 1000
# CODE
params = {
    "lr": lr,
    "bs": bs,
    "save_dir": save_dir,
    "dataset_increment": dataset_increment,
    "dataset_initial_length": dataset_initial_length,
}
run["parameters"] = params
os.makedirs(save_dir, exist_ok=True)
model.train()
global_counter = 0
threshold = 0.1
for n_increase_of_dataset in range(1000):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = ExponentialLR(optimizer, 0.99)
    size_of_dataset = dataset_initial_length + n_increase_of_dataset * dataset_increment
    dataset.set_length(size_of_dataset)
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [int(0.9 * size_of_dataset), int(0.1 * size_of_dataset)]
    )
    for n_epoch in tqdm(range(n_epochs), desc="Epoch", leave=True):
        model.to("cuda")
        train_avg_loss = 0
        n_iters = 0
        train_dataloader = DataLoader(train_dataset, bs, shuffle=True)
        for sample in train_dataloader:
            model.train()
            # Get data
            (img1, img2), labels = sample
            img1 = img1.to("cuda")
            img2 = img2.to("cuda")
            noise1 = torch.tensor(np.random.normal(size=img1.shape) * 2 / 100 + 1).to("cuda")
            noise2 = torch.tensor(np.random.normal(size=img2.shape) * 2 / 100 + 1).to("cuda")
            img1 *= noise1
            img2 *= noise2
            labels = labels.to("cuda")
            # Forward pass
            optimizer.zero_grad()
            result = model(img1, img2)
            # calculate loss
            train_loss = criterion1(result, labels)
            run[f"train/loss/{size_of_dataset}"].append(train_loss.item())
            train_avg_loss += train_loss.item()
            n_iters += 1
            train_loss.backward()
            optimizer.step()
            ## test data
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
            run[f"test/loss/{size_of_dataset}"].append(test_loss.item())
            global_counter += 1

        if train_avg_loss / n_iters < threshold:
            torch.save(model.to("cpu").state_dict(), os.path.join(save_dir, f"last.torch"))
            break
        lr_scheduler.step()
run.stop()
