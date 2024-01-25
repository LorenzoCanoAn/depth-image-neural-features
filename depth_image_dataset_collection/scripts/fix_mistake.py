import numpy  as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

folder_to_fix = "/home/lorenzo/.datasets/comprehensive_depth_image_dataset"
for file in os.listdir(folder_to_fix):
    path_to_images_file = os.path.join(folder_to_fix, file)
    if not ".npy" in path_to_images_file:
        continue
    path_to_save_folder = os.path.join(folder_to_fix, file.replace(".npy", ""))
    os.makedirs(path_to_save_folder, exist_ok=True)
    print(f"Loading: {path_to_images_file}")
    images = np.load(path_to_images_file)
    n_images = len(images)
    for n_data, image in tqdm(enumerate(images), total=n_images):
        assert isinstance(image, np.ndarray)
        image_file_name = f"{n_data:010d}"
        path_to_image_file = os.path.join(path_to_save_folder, image_file_name)
        print(path_to_image_file)
        plt.imshow(image) 
        plt.show()
        break