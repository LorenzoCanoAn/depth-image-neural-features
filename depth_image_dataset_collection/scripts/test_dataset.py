import json
from tqdm import tqdm
import numpy as np
import torch
import os
import pyvista as pv
from pyvista.plotting.plotter import Plotter


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
        self.pairs_index = [["world", 0, 0, np.zeros((0, 0))] for _ in range(n_pairs)]
        n_pair = 0
        for world_name in tqdm(self.data.keys(), total=len(list(self.data.keys()))):
            n_pairs_for_this_world = n_pairs_per_world[world_name]
            points_of_this_world = self.data[world_name]["poses"]
            idxs1 = random_non_repeat_ints(
                len(points_of_this_world), n_pairs_for_this_world
            )
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
                        self.pairs_index[n_pair][3] = torch.tensor(
                            vector[:2].astype(np.float32)
                        )
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
                np.load(
                    os.path.join(self.data[wn]["images_folder_path"], f"{idx:010d}.npy")
                )
            )

    def __len__(self):
        return self.n_pairs

    def __getitem__(self, idx):
        wn, id1, id2, v = self.pairs_index[idx]
        img1 = self.images[self.images_dict[(wn, id1)]]
        img2 = self.images[self.images_dict[(wn, id2)]]
        return (img1, img2), v


dataset = IndexBasedDataset(
    "/home/lorenzo/.datasets/comprehensive_depth_image_dataset/index.json", 3
)
print(dataset.index["info"].keys())
dataset.generate_pairs(300)


def img_to_ptcl(img, max_dist):
    img_ = 1 - img
    img_[np.where(img == 0)] = 0
    img_ = img_ * max_dist
    distances = np.reshape(img_, -1)
    img_ones = np.ones(img.shape)
    thetas_raw = np.reshape(np.linspace(np.deg2rad(15), np.deg2rad(-15), 16), (16, 1))
    phis_raw = np.reshape(np.linspace(0, np.pi * 2, 1024), (1, 1024))
    thetas = np.reshape(thetas_raw * img_ones, -1)
    phis = np.reshape(phis_raw * img_ones, -1)
    x = distances * np.cos(phis) * np.cos(thetas)
    y = distances * np.sin(phis) * np.cos(thetas)
    z = distances * np.sin(thetas)
    ptcl = np.vstack((x, y, z))
    return ptcl.T


for i in range(len(dataset)):
    (img1, img2), v = dataset[i]
    img1 = np.array(img1)
    img2 = np.array(img2)
    ptcl1 = img_to_ptcl(img1, dataset.index["info"]["max_distance"])
    ptcl2 = img_to_ptcl(img2, dataset.index["info"]["max_distance"])
    plotter = Plotter()
    plotter.add_mesh(pv.PolyData(ptcl1), color="b")
    plotter.add_mesh(pv.PolyData(ptcl2), color="r")
    plotter.add_axes_at_origin()
    plotter.show()
