from dataset_management.dataset import DatasetFileManagerToPytorchDataset
import math
import numpy as np
from tqdm import tqdm
import torch
import random
from tqdm.contrib.concurrent import process_map  # or thread_map


def euler_to_quaternion(yaw, pitch, roll):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)
    return [qx, qy, qz, qw]


def T_to_xyzrpy(T):
    translation = T[:3, 3]
    x, y, z = translation
    rotation_matrix = T[:3, :3]
    pitch = -math.asin(rotation_matrix[2, 0])
    if math.cos(pitch) != 0:
        yaw = math.atan2(
            rotation_matrix[1, 0] / math.cos(pitch),
            rotation_matrix[0, 0] / math.cos(pitch),
        )
    else:
        yaw = 0
    roll = math.atan2(
        rotation_matrix[2, 1] / math.cos(pitch), rotation_matrix[2, 2] / math.cos(pitch)
    )
    pitch = math.degrees(pitch)
    yaw = math.degrees(yaw)
    roll = math.degrees(roll)
    return x, y, z, roll, pitch, yaw


def xyzrpyw_to_T(x, y, z, roll, pitch, yaw):
    translation_matrix = np.array(
        [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]]
    )
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    rotation_yaw = np.array(
        [
            [cos_yaw, -sin_yaw, 0, 0],
            [sin_yaw, cos_yaw, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)
    rotation_pitch = np.array(
        [
            [1, 0, 0, 0],
            [0, cos_pitch, -sin_pitch, 0],
            [0, sin_pitch, cos_pitch, 0],
            [0, 0, 0, 1],
        ]
    )
    cos_roll = math.cos(roll)
    sin_roll = math.sin(roll)
    rotation_roll = np.array(
        [
            [cos_roll, 0, sin_roll, 0],
            [0, 1, 0, 0],
            [-sin_roll, 0, cos_roll, 0],
            [0, 0, 0, 1],
        ]
    )
    rotation_matrix = np.dot(np.dot(rotation_yaw, rotation_pitch), rotation_roll)
    transformation_matrix = np.dot(translation_matrix, rotation_matrix)
    return transformation_matrix


def gen_label(transform1, transform2):
    x1, y1, z1, r1, p1, yw1 = np.reshape(np.array(transform1), -1)
    x2, y2, z2, r2, p2, yw2 = np.reshape(np.array(transform2), -1)
    T1 = xyzrpyw_to_T(x1, y1, z1, r1, p1, yw1)
    T2 = xyzrpyw_to_T(x2, y2, z2, r2, p2, yw2)
    T12 = np.dot(np.linalg.inv(T1), T2)
    x, y, z, roll, pitch, yaw = T_to_xyzrpy(T12)
    qx, qy, qz, qw = euler_to_quaternion(yaw, pitch, roll)
    return x, y, z, qx, qy, qz, qw


class DepthImageDistanceFeaturesDataset(DatasetFileManagerToPytorchDataset):
    required_identifiers = [
        "height",
        "width",
        "max_distance",
        "inverted_distance",
        "n_stackings",
        "normalized_image",
    ]

    def __init__(
        self,
        name=None,
        mode="read",
        identifiers=dict(),
        unwanted_characteristics=dict(),
        samples_to_generate=100,
        prob_of_same_pose_sample=0.5,
    ):
        super().__init__(
            name,
            mode,
            identifiers,
            unwanted_characteristics,
            samples_to_generate=samples_to_generate,
            prob_of_same_pose_sample=prob_of_same_pose_sample,
        )

    def process_raw_inputs(self):
        # _loaded_inputs is a torch array of N images in grayscale
        # _loaded_labels is a torch array of N vectors of x,y,z,roll,pitch,yaw,environment number,and code
        # In the same dataset there can be more than one env -> it is necessary to create the labels env per env
        n_datapoints_per_datafolder = [
            datafolder.n_files for datafolder in self.input_manager.selected_datafolders
        ]
        total_datapoints = len(self._loaded_labels)
        if total_datapoints == self.samples_to_generate:
            samples_to_generate_per_datafolder = n_datapoints_per_datafolder
        else:
            samples_to_generate_per_datafolder = []
            for n_datapoints in n_datapoints_per_datafolder:
                samples_to_generate_per_datafolder.append(
                    int(n_datapoints / total_datapoints * self.samples_to_generate)
                )
            if sum(samples_to_generate_per_datafolder) != self.samples_to_generate:
                samples_to_generate_per_datafolder[
                    -1
                ] += self.samples_to_generate - sum(samples_to_generate_per_datafolder)
        self.start_idx = 0
        self._inputs = [None for _ in range(self.samples_to_generate)]
        self._labels = [None for _ in range(self.samples_to_generate)]
        for n_datafolder, n_samples_to_generate in tqdm(
            enumerate(samples_to_generate_per_datafolder), desc="Env progression"
        ):
            self.start_idx_for_loaded = sum(n_datapoints_per_datafolder[:n_datafolder])
            self.end_idx_for_loaded = sum(
                n_datapoints_per_datafolder[: n_datafolder + 1]
            )

            for n_sample in tqdm(range(n_samples_to_generate), desc="Gen samples"):
                self.calc_sample(n_sample)
            self.start_idx += n_samples_to_generate

    def calc_sample(self, n_sample):
        while True:
            idx1 = random.randint(
                self.start_idx_for_loaded, self.end_idx_for_loaded - 1
            )
            idx2 = random.randint(
                self.start_idx_for_loaded, self.end_idx_for_loaded - 1
            )
            img1 = self._loaded_inputs[idx1]
            img2 = self._loaded_inputs[idx2]
            self._inputs[self.start_idx + n_sample] = (img1, img2)
            distance = gen_label(self._loaded_labels[idx1], self._loaded_labels[idx2])
            self._labels[self.start_idx + n_sample] = torch.Tensor(distance)
            if distance[0] < 1.5:
                break

    def import_args(self, samples_to_generate, prob_of_same_pose_sample):
        self.samples_to_generate = samples_to_generate
        self.prob_of_same_pose_sample = prob_of_same_pose_sample


def fold_image(image, n_folds):
    height, width = image.shape
    assert width % 2**n_folds == 0
    for n_stack in range(1, n_folds + 1):
        new_width = int(width / 2**n_stack)
        image = np.vstack((image[:, :new_width], image[:, new_width:]))
    return image


def unfold_image(image, n_folds):
    height, width = image.shape
    assert height % 2**n_folds == 0
    for n_stack in range(1, n_folds + 1):
        new_height = int(height / 2**n_stack)
        image = np.hstack((image[:new_height, :], image[:, new_height:]))
    return image
