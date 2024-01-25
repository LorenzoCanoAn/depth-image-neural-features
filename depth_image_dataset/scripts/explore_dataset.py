import matplotlib.pyplot as plt
import numpy as np
import torch
import os

PARAMETERS = {
    'base_folder':"/home/lorenzo/.datasets/depth_image_feature_extraction/0"
}


def main(PARAMETERS): 
    folder = PARAMETERS["base_folder"]
    file_names = os.listdir(folder)
    path_to_file = os.path.join(folder,file_names[0])
    data = np.load(path_to_file)
    input_data = data["input_data"][0]
    print(np.max(input_data))
    plt.imshow(input_data)
    plt.show()

if __name__ == "__main__":
    main(PARAMETERS)