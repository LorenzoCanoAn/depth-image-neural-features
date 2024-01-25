import json
import os

PARAMETERS = {
    "dataset_folder":"/home/lorenzo/.datasets/comprehensive_depth_image_dataset"
}

def main(PARAMETERS):
    folder_of_dataset = PARAMETERS["dataset_folder"]
    path_to_index = os.path.join(folder_of_dataset, 'index.json')
    assert os.path.isfile(path_to_index)
    with open(path_to_index, "r") as f:
        index = json.load(f)
    data = index['data']
    with open("info.txt", "w+") as f:
        for world_name in data.keys():
            world_data = data[world_name]
            images_folder = world_data["images_folder_path"]
            for i in range(len(world_data["poses"])):
                file_name = f"{i:010d}.npy"
                path_to_file = os.path.join(images_folder,file_name)
                assert os.path.isfile(path_to_file)
                f.write(path_to_file)
                f.write("\n")
    

if __name__ == "__main__":
    main(PARAMETERS)