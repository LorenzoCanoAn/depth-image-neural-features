import roslaunch
import os

def launch_file_by_args(uuid, cli_args):
    if len(cli_args)>2:
        roslaunch_args = cli_args[2:]    
        roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]
    else:
        roslaunch_file = roslaunch.rlutil.resolve_launch_arguments(cli_args) 
    return roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file)

def setup_env_variables():
    os.environ["HUSKY_STATIC"] = "1"
    os.environ["HUSKY_LASER_3D_ENABLED"] = "1"

def launch_env(uuid, path_to_world):
    gazebo_args = ['gazebo_ros', 'empty_world.launch', f'world_name:={path_to_world}']
    husky_args = ['husky_gazebo', 'spawn_husky.launch']
    gazebo_parent = launch_file_by_args(uuid, gazebo_args)
    husky_parent = launch_file_by_args(uuid, husky_args)
    gazebo_parent.start()
    husky_parent.start()
    return gazebo_parent, husky_parent
    
def list_to_string(my_list):
    my_list_string = ""
    for n, element in enumerate(my_list):
        if n != 0:
            my_list_string +=","
        my_list_string += element
    return my_list_string

def main():
    path_to_selected_worlds_file = "/home/lorenzo/world_index.txt"
    worlds_folder = "/home/lorenzo/repos/gazebo_models_worlds_collection/worlds"
    world_paths = os.listdir(worlds_folder)
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    setup_env_variables()
    selected_worlds = []
    if os.path.isfile(path_to_selected_worlds_file):
        with open(path_to_selected_worlds_file, "r") as f:
            selected_worlds_string = f.read()
        selected_worlds = selected_worlds_string.split(",")
    for world_path in world_paths:
        if world_path in selected_worlds:
            continue
        parents = launch_env(uuid, world_path)
        print(world_path)
        inpt = input()
        if inpt.lower() == "y":
            selected_worlds.append(world_path)
            with open(path_to_selected_worlds_file, "w+") as f:
                selected_worlds_string = list_to_string(selected_worlds)
                f.write(selected_worlds_string)
        else:
            os.remove(os.path.join(worlds_folder, world_path))
        for p in parents[::-1]:
            p.shutdown()
if __name__ == "__main__":
    main()