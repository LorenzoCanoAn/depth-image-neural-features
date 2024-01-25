import os
import numpy as np
import shutil
import matplotlib.pyplot as plt
import json
import shapely
import random
import math
from tqdm import tqdm
import roslaunch
import rospy
import time
from gazebo_msgs.srv import SetModelState, SetModelStateRequest, SetModelStateResponse
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import threading

PARAMETERS = {
    "density": 20,
    "image_height": 16,
    "image_width": 1024,
    "max_distance": 50,
    "invert_distance": "true",
    "normalize_image": "true",
    "roll_range": np.deg2rad(3),
    "pitch_range": np.deg2rad(3),
    "path_to_worlds_folder": "/home/lorenzo/gazebo_worlds/comprehensive_dataset_worlds",
    "path_to_dataset": "/home/lorenzo/.datasets/comprehensive_depth_image_dataset",
}


def sample_points_inside_polygon(polygon, n):
    min_x, min_y, max_x, max_y = polygon.bounds
    sampled_points = np.zeros((n, 2))
    pcounter = 0
    while pcounter < n:
        random_point = shapely.Point(
            random.uniform(min_x, max_x), random.uniform(min_y, max_y)
        )
        if polygon.contains(random_point):
            sampled_points[pcounter, :] = np.array((random_point.x, random_point.y))
            pcounter += 1
    return sampled_points


def rectangle_from_segment_and_width(p1, p2, width):
    p1 = np.array(p1, dtype=float)[:2]  # Contains xyz but we only want xy
    p2 = np.array(p2, dtype=float)[:2]
    vector12 = p2 - p1
    vector12 /= np.linalg.norm(vector12, 2)
    rotation_matrix = np.array(
        [
            [np.cos(np.pi / 2), -np.sin(np.pi / 2)],
            [np.sin(np.pi / 2), np.cos(np.pi / 2)],
        ]
    )
    vectorp = np.dot(rotation_matrix, vector12)
    sp1 = p1 + vectorp * width / 2
    sp2 = p1 - vectorp * width / 2
    sp3 = p2 - vectorp * width / 2
    sp4 = p2 + vectorp * width / 2
    return (sp1, sp2, sp3, sp4)


class PolygonArea:
    def __init__(self, raw_data):
        self.points = raw_data["points"]
        self.n_points = len(self.points)
        self.polygon = shapely.Polygon(self.points)
        self.total_area = self.polygon.area

    def sample_n_points(self, n_points):
        return sample_points_inside_polygon(self.polygon, n_points)

    def sample_points_by_density(self, density):
        n_points = int(self.total_area * density)
        return self.sample_n_points(n_points)


class PolylineArea:
    def __init__(self, raw_data):
        self.points = raw_data["points"]
        self.n_points = len(self.points)
        self.width = raw_data["width"]
        self.polygons = []
        for i in range(self.n_points - 1):
            p1 = self.points[i]
            p2 = self.points[i + 1]
            rectangle_points = rectangle_from_segment_and_width(p1, p2, self.width)
            self.polygons.append(shapely.Polygon(rectangle_points))
        self.polygon_areas = [p.area for p in self.polygons]
        self.total_area = sum(self.polygon_areas)

    def sample_n_points(self, n_points):
        n_points_per_polygon = [
            int(math.floor(n_points * area / self.total_area))
            for area in self.polygon_areas
        ]
        q = 0
        while sum(n_points_per_polygon) < n_points:
            n_points_per_polygon[q] += 1
            q += 1
            q = q % len(n_points_per_polygon)
        sampled_points = np.zeros((0, 2))
        for i in range(len(self.polygons)):
            polygon = self.polygons[i]
            n_points = n_points_per_polygon[i]
            sampled_points = np.vstack(
                (sampled_points, sample_points_inside_polygon(polygon, n_points))
            )
        return sampled_points

    def sample_points_by_density(self, density):
        n_points = int(self.total_area * density)
        return self.sample_n_points(n_points)


class Area:
    def __init__(self, raw_data):
        self.area_type = raw_data["type"]
        if self.area_type == "polygon":
            self.geometry_manager = PolygonArea(raw_data)
        elif self.area_type == "polyline":
            self.geometry_manager = PolylineArea(raw_data)
        else:
            raise Exception(f"Area of type {self.area_type} does not exist")

    @property
    def total_area(self):
        return self.geometry_manager.total_area

    def sample_points_by_density(self, density):
        return self.geometry_manager.sample_points_by_density(density)

    def sample_n_points(self, n_points):
        return self.geometry_manager.sample_n_points(n_points)


class AreaManager:
    # This interacts with the areas file
    def __init__(self, path_to_areas_file):
        self.path_to_file = path_to_areas_file
        with open(self.path_to_file, "r") as f:
            self.json_areas = json.load(f)
        self.areas = [Area(json_areas) for json_areas in self.json_areas]
        self.area_of_areas = [area.total_area for area in self.areas]
        self.total_area = sum(self.area_of_areas)

    def sample_n_points(self, n_points):
        n_points_per_area = [
            int(math.floor(n_points * area / self.total_area))
            for area in self.area_of_areas
        ]
        q = 0
        while sum(n_points_per_area) < n_points:
            n_points_per_area[q] += 1
            q += 1
            q = q % len(n_points_per_area)
        sampled_points = np.zeros((0, 2))
        for n_area, area in enumerate(self.areas):
            n_points = n_points_per_area[n_area]
            sampled_points = np.vstack((sampled_points, area.sample_n_points(n_points)))
        return sampled_points

    def sample_points_by_density(self, density):
        sampled_points = np.zeros((0, 2))
        for area in self.areas:
            sampled_points = np.vstack(
                (sampled_points, area.sample_points_by_density(density))
            )
        return sampled_points


class WorldsManager:
    # This interacts with the world folder
    def __init__(self, base_folder):
        self.base_folder = base_folder
        self.world_names = os.listdir(self.base_folder)
        self.world_folder_paths = [
            os.path.join(self.base_folder, world_folder)
            for world_folder in self.world_names
        ]
        self.world_file_paths = [
            os.path.join(world_folder_path, "world.world")
            for world_folder_path in self.world_folder_paths
        ]
        self.areas_file_paths = [
            os.path.join(world_folder_path, "areas.json")
            for world_folder_path in self.world_folder_paths
        ]

    def __len__(self):
        return len(self.world_names)

    def __getitem__(self, idx):
        return (
            self.world_names[idx],
            self.world_folder_paths[idx],
            self.world_file_paths[idx],
            self.areas_file_paths[idx],
        )

    def __iter__(self):
        self.iter_counter = 0
        return self

    def __next__(self):
        if self.iter_counter == self.__len__():
            raise StopIteration()
        to_return = self[self.iter_counter]
        self.iter_counter += 1
        return to_return


def launch_file_by_args(uuid, cli_args, supress_output=True):
    if len(cli_args) > 2:
        roslaunch_args = cli_args[2:]
        roslaunch_file = [
            (roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)
        ]
    else:
        roslaunch_file = roslaunch.rlutil.resolve_launch_arguments(cli_args)
    return roslaunch.parent.ROSLaunchParent(
        uuid, roslaunch_file, force_log=supress_output
    )


class ImageTopicStorage:
    def __init__(self, topic_name, time_to_sleep=0.05):
        print(f"created topic storage for topic {topic_name}")
        self.bridge = CvBridge()
        rospy.Subscriber(topic_name, Image, callback=self.callback)
        self.time_to_sleep = time_to_sleep
        self.msg_recieved = False
        self.last_msg = None

    def callback(self, msg):
        self.msg_recieved = True
        self.last_msg = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")

    def block(self, n_msgs=1):
        for _ in range(n_msgs):
            self.msg_recieved = False
            while not self.msg_recieved:
                time.sleep(self.time_to_sleep)
        return self.last_msg


def euler_to_quaternion(roll, pitch, yaw):
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
    return qx, qy, qz, qw


class DatasetCaptureNode:
    def __init__(self, index: dict, retake_all):
        self.retake_all = retake_all
        self.index = index
        self.width = index["info"]["width"]
        self.height = index["info"]["height"]
        self.max_distance=index["info"]["max_distance"] 
        self.invert_distance=index["info"]["invert_distance"]
        self.normalize_image=index["info"]["normalize_image"]
        self.data = index["data"]

    def capture_dataset(self):
        self.setup_env_variables()
        rospy.init_node("dataset_capture")
        ros_thread = threading.Thread(target=self.ros_spin_target)
        ros_thread.start()
        self.move_robot_service_proxy = rospy.ServiceProxy(
            "/gazebo/set_model_state", SetModelState
        )
        if self.retake_all:
            for key in self.data.keys():
                world_data = self.data[key]
                images_folder_path = world_data["images_folder_path"]
                if os.path.isdir(images_folder_path):
                    shutil.rmtree(images_folder_path)
        for key in self.data.keys():
            world_data = self.data[key]
            path_to_world_file = world_data["world_file_path"]
            images_folder_path = world_data["images_folder_path"]
            poses = world_data["poses"]
            if os.path.isdir(images_folder_path):
                start_from = len(os.listdir(images_folder_path))
            else:
                start_from = 0
            if start_from == len(poses):
                continue
            self.ros_startup(path_to_world_file)
            self.capture_images(poses, images_folder_path, start_from =start_from)
            self.ros_shutdown()

    def move_robot(self, pose):
        x, y, z, roll, pitch, yaw = pose
        qx, qy, qz, qw = euler_to_quaternion(roll, pitch, yaw)
        rqst = SetModelStateRequest()
        rqst.model_state.model_name = "husky"
        rqst.model_state.reference_frame = ""
        rqst.model_state.pose.position.x = x
        rqst.model_state.pose.position.y = y
        rqst.model_state.pose.position.z = z
        rqst.model_state.pose.orientation.x = qx
        rqst.model_state.pose.orientation.y = qy
        rqst.model_state.pose.orientation.z = qz
        rqst.model_state.pose.orientation.w = qw
        while True:
            try:
                self.move_robot_service_proxy.call(rqst)
                break
            except:
                pass

    def ros_spin_target(self):
        rospy.spin()
    
    def capture_images(self, poses, images_folder_path, start_from = 0):
        os.makedirs(images_folder_path, exist_ok=True)
        image_storage = ImageTopicStorage("/depth_image")
        n_poses = len(poses)
        print("Starting from: ", start_from)
        for n_pose in tqdm(range(start_from, n_poses),initial=start_from, total=n_poses):
            pose = poses[n_pose]
            self.move_robot(pose)
            image = image_storage.block(3)
            image_file_name = f"{n_pose:010d}.npy"
            path_to_image = os.path.join(images_folder_path, image_file_name)
            with open(path_to_image, "wb+") as f:
                np.save(f, image)
            

    def setup_env_variables(self):
        os.environ["HUSKY_STATIC"] = "1"
        os.environ["HUSKY_LASER_3D_ENABLED"] = "1"

    def ros_startup(self, path_to_world_file):
        self.roslaunch_parents = []
        self.start_gazebo(path_to_world_file)
        self.start_lidar_to_img()

    def ros_shutdown(self):
        for parent in self.roslaunch_parents:
            parent.shutdown()

    def start_gazebo(self, path_to_world_file):
        self.uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        gazebo_args = [
            "gazebo_ros",
            "empty_world.launch",
            f"world_name:={path_to_world_file}",
        ]
        husky_args = ["husky_gazebo", "spawn_husky.launch"]
        gazebo_parent = launch_file_by_args(self.uuid, gazebo_args)
        husky_parent = launch_file_by_args(self.uuid, husky_args)
        print("Launching gazebo")
        gazebo_parent.start()
        time.sleep(30)
        rospy.wait_for_service("/gazebo/set_model_state")
        print("Spawning husky")
        husky_parent.start()
        time.sleep(5)
        self.roslaunch_parents.append(gazebo_parent)
        self.roslaunch_parents.append(husky_parent)

    def start_lidar_to_img(self):
        lidar_to_img_args = [
            "lidar_to_other",
            "pointcloud_to_depth_image.launch",
            f"height:={self.height}",
            f"width:={self.width}",
            f"max_distance:={self.max_distance}",
            f"invert_distance:={self.invert_distance}",
            f"normalize_image:={self.normalize_image}",
            "void_value:=0",
        ]
        lidar_to_img_parent = launch_file_by_args(self.uuid, lidar_to_img_args,supress_output=False)
        lidar_to_img_parent.start()
        time.sleep(1)
        self.roslaunch_parents.append(lidar_to_img_parent)


def main(PARAMETERS):
    # Extract the parameters
    path_to_worlds_folder = PARAMETERS["path_to_worlds_folder"]
    path_to_dataset = PARAMETERS["path_to_dataset"]
    density = PARAMETERS["density"]
    image_width = PARAMETERS["image_width"]
    image_height = PARAMETERS["image_height"]
    roll_range = PARAMETERS["roll_range"]
    pitch_range = PARAMETERS["pitch_range"]
    max_distance = PARAMETERS["max_distance"]
    invert_distance = PARAMETERS["invert_distance"]
    normalize_image = PARAMETERS["normalize_image"]
    # Generate derived parameters
    path_to_dataset_index_file = os.path.join(path_to_dataset, "index.json")
    do_poses_generation = False
    if os.path.isfile(path_to_dataset_index_file):
        inpt = input(f"The poses have already been generated, generate_again? [yes, generate again]: ")
        if inpt.lower() == "yes, generate again":
            inpt = input(f"Are you sure? [yes, i am sure]: ")
            if inpt.lower() == "yes, i am sure":
                do_poses_generation = True
    if do_poses_generation:
        worlds_manager = WorldsManager(path_to_worlds_folder)
        index = {
            "info": {
                "density": density,
                "width": image_width,
                "height": image_height,
                "pitch_range": pitch_range,
                "roll_range": roll_range,
                "max_distance": max_distance,
                "invert_distance": invert_distance,
                "normalize_image": normalize_image,
            },
            "data": {},
        }  # Each entrance in the data entry is a np.array of shape Nx6, where each column is xyzrpy
        # Generate the poses
        for world_name, world_folder, world_file_path, areas_file_path in tqdm(
            worlds_manager, total=len(worlds_manager)
        ):
            areas_manager = AreaManager(areas_file_path)
            positions = areas_manager.sample_points_by_density(density)
            n_positions = len(positions)
            zs = np.random.uniform(0.14, 0.19, [n_positions, 1])
            rolls = np.random.uniform(-roll_range, roll_range, [n_positions, 1])
            pitches = np.random.uniform(-pitch_range, pitch_range, [n_positions, 1])
            yaws = np.random.uniform(0, 2 * np.pi, [n_positions, 1])
            poses = np.concatenate((positions,zs, rolls, pitches, yaws), axis=1)
            images_folder_path = os.path.join(path_to_dataset, world_name)
            index["data"][world_name] = {
                "world_name": world_name,
                "poses": poses.tolist(),
                "world_file_path": world_file_path,
                "images_folder_path": images_folder_path,
            }
        with open(path_to_dataset_index_file, "w+") as f:
            json.dump(index, f)
    else:
        with open(path_to_dataset_index_file, "r") as f:
            print("Loading index file")
            index = json.load(f)
            print("Index loaded")
    # START THE DATASET CAPTURE
    node = DatasetCaptureNode(index, do_poses_generation)
    node.capture_dataset()

if __name__ == "__main__":
    main(PARAMETERS)
