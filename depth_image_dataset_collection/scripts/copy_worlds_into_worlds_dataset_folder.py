import os

def create_folder_and_move_world(path_to_world, destination_folder):
    world_name = os.path.split(path_to_world)[-1].replace(".world","")
    world_folder = os.path.join(destination_folder,world_name)
    os.makedirs(world_folder, exist_ok=True)
    new_world_file_path = os.path.join(world_folder, "world.world")
    command = f"ln -s {path_to_world} {new_world_file_path}"
    os.system(command)
    
def main():
    folder_with_worlds_to_use = "/home/lorenzo/repos/gazebo_models_worlds_collection/worlds"
    folder_for_dataset_worlds = "/home/lorenzo/gazebo_worlds/comprehensive_dataset_worlds"
    assert os.path.isdir(folder_with_worlds_to_use)
    assert os.path.isdir(folder_for_dataset_worlds)
    world_paths = [os.path.join(folder_with_worlds_to_use, world_file_name) for world_file_name in os.listdir(folder_with_worlds_to_use)]
    for world_path in world_paths:
        assert os.path.isfile(world_path)
        create_folder_and_move_world(world_path, folder_for_dataset_worlds)
   
if __name__ == "__main__" :
    main()