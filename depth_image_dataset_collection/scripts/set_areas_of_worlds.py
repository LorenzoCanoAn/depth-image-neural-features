import roslaunch
import os
import time

PARAMETERS = {
    "folder_of_worlds": "/home/lorenzo/gazebo_worlds/comprehensive_dataset_worlds"
}


##################################################################################################################
#   LAUNCH FUNCTIONS
##################################################################################################################
def launch_file_by_args(uuid, cli_args):
    if len(cli_args) > 2:
        roslaunch_args = cli_args[2:]
        roslaunch_file = [
            (roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)
        ]
    else:
        roslaunch_file = roslaunch.rlutil.resolve_launch_arguments(cli_args)
    return roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file)


def launch_env(uuid, path_to_world):
    gazebo_args = ["gazebo_ros", "empty_world.launch", f"world_name:={path_to_world}"]
    gazebo_parent = launch_file_by_args(uuid, gazebo_args)
    gazebo_parent.start()
    return gazebo_parent

def launch_area_collection_node():
        package = "depth_image_dataset_collection"
        executable = "collect_world_areas_node.py"
        node = roslaunch.core.Node(package, executable, output="screen")
        launch = roslaunch.scriptapi.ROSLaunch()
        launch.start()
        process = launch.launch(node)
        while process.is_alive():
            time.sleep(0.1)
            
def set_ros_parameter(parameter_name, value):
    os.system(f"rosparam set {parameter_name} {value}")
            
def main(PARAMETERS):
    base_folder = PARAMETERS["folder_of_worlds"]
    world_folders = [
        os.path.join(base_folder, world_name) for world_name in os.listdir(base_folder)
    ]
    for world_folder in world_folders:
        if os.path.isfile(os.path.join(world_folder, "areas.json")):
            continue
            inpt = input(f"There is an areas file for this the world {world_folder}, retake area? [y/n]: ")
            if inpt.lower() != "y":
                continue
        print(world_folder)
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        path_to_world = os.path.join(world_folder, "world.world")
        env_parent = launch_env(uuid, path_to_world)
        set_ros_parameter("/area_collection/folder_of_current_world", world_folder)
        launch_area_collection_node()
        env_parent.shutdown()


if __name__ == "__main__":
    main(PARAMETERS)
