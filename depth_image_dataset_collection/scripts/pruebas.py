import numpy as np
import os
import matplotlib.pyplot as plt
folder = "/home/lorenzo/.datasets/comprehensive_depth_image_dataset"

for element in os.listdir(folder):
    path_to_element = os.path.join(folder, element)
    is_dir = os.path.isdir(path_to_element)
    if not is_dir:
        continue
    for datapoint_file_name in os.listdir(path_to_element):
        path_to_datapoint = os.path.join(path_to_element, datapoint_file_name)
        image = np.load(path_to_datapoint)
        plt.imshow(image)
        plt.show()
        