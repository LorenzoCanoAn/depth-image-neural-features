#!/usr/bin/python
import rospy
from gazebo_msgs.srv import SetModelState, SetModelStateResponse, SetModelStateRequest
from depth_image_dataset.dataset import DepthImageDistanceFeaturesDataset
from dataset_management.dataset_io import DataFoldersManager
import json
import os
import threading
import math
import numpy as np
from tqdm import tqdm
from shapely import Polygon, Point
import time
from sensor_msgs.msg import Image
import cv2
import cv_bridge


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


def quaternion_to_euler(q):
    (x, y, z, w) = (q[0], q[1], q[2], q[3])
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return [yaw, pitch, roll]


def gen_pose_from_free_space(free_space_info) -> (Point, Point):
    # TODO: what hapens if there are more than one free space
    for free_space in free_space_info:
        if free_space["type"] == "polygon":
            return random_points_in_polygon(
                Polygon(np.array(free_space["points"])[:, :2]), 1
            )[0]


def random_points_in_polygon(polygon: Polygon, number: int):
    points = []
    minx, miny, maxx, maxy = polygon.bounds
    while len(points) < number:
        pnt = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        if polygon.contains(pnt):
            points.append(pnt)
    return points


class ImageTopicStorage:
    def __init__(self, topic_name, time_to_sleep=0.05):
        print(f"created topic storage for topic {topic_name}")
        self.bridge = cv_bridge.CvBridge()
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


def stack_image(image, n_stackings):
    height, width = image.shape
    assert width % 2**n_stackings == 0
    for n_stack in range(1, n_stackings + 1):
        new_width = int(width / 2**n_stack)
        image = np.vstack((image[:, :new_width], image[:, new_width:]))
    return image

dataset_type = DepthImageDistanceFeaturesDataset


class DatasetCollectionNode:
    def __init__(self):
        rospy.init_node("dataset_collection_node")
        # Get params
        # -----------------------------------------------------------------------
        self.paths_to_envs = rospy.get_param("~paths_to_envs").split(",")
        self.n_poses_per_env = rospy.get_param("~n_poses_per_env")
        self.n_orientations_per_pose = rospy.get_param("~n_orientations_per_pose")
        self.robot_name = rospy.get_param("~robot_name")
        self.image_topic = rospy.get_param("~image_topic")
        self.dataset_name = rospy.get_param("~dataset_name")
        self.n_stackings = rospy.get_param("~n_stackings")
        self.height = rospy.get_param("~height")
        self.width = rospy.get_param("~width")
        self.max_distance = rospy.get_param("~max_distance")
        self.invert_distance = rospy.get_param("~invert_distance")
        self.normalize_image = rospy.get_param("~normalize_image")
        self.void_value = rospy.get_param("~void_value")
        self.max_incl = rospy.get_param("~max_incl")
        identifiers = dict()
        for necessary_key in dataset_type.required_identifiers:
            identifiers[necessary_key] = rospy.get_param("~" + necessary_key)
        identifiers["n_stackings"] = self.n_stackings
        identifiers["height"] = self.height
        identifiers["width"] = self.width
        identifiers["max_distance"] = self.max_distance
        identifiers["invert_distance"] = self.invert_distance
        identifiers["normalize_image"] = self.normalize_image
        identifiers["void_value"] = self.void_value
        identifiers["max_incl"] = self.max_incl
        # -----------------------------------------------------------------------
        self.ros_thread = threading.Thread(target=self.ros_thread_target)
        self.dataset = dataset_type(
            name=self.dataset_name, mode="write", identifiers=identifiers
        )
        self.move_robot_service_proxy = rospy.ServiceProxy(
            "/gazebo/set_model_state", SetModelState
        )
        self.image_storage = ImageTopicStorage(self.image_topic)

    def move_robot(self, *args):
        if len(args) == 1:
            x, y, z, roll, pitch, yaw = args[0]
        qx, qy, qz, qw = euler_to_quaternion(roll, pitch, yaw)
        rqst = SetModelStateRequest()
        rqst.model_state.model_name = self.robot_name
        rqst.model_state.reference_frame = ""
        rqst.model_state.pose.position.x = x
        rqst.model_state.pose.position.y = y
        rqst.model_state.pose.position.z = z
        rqst.model_state.pose.orientation.x = qx
        rqst.model_state.pose.orientation.y = qy
        rqst.model_state.pose.orientation.z = qz
        rqst.model_state.pose.orientation.w = qw
        self.move_robot_service_proxy.call(rqst)

    def ros_thread_target(self):
        rospy.spin()

    def run(self):
        self.ros_thread.start()
        for path_to_env in tqdm(self.paths_to_envs):
            self.dataset.new_env(path_to_env)
            free_space_file = os.path.join(path_to_env, "free_space.json")
            with open(free_space_file, "r") as f:
                free_space_info = json.load(f)
            for _ in tqdm(range(self.n_poses_per_env)):
                xy1 = gen_pose_from_free_space(free_space_info)
                for _ in range(self.n_orientations_per_pose):
                    pose = (
                        xy1.x,
                        xy1.y,
                        0.18,
                        np.random.uniform(np.deg2rad(-self.max_incl),np.deg2rad(self.max_incl)),
                        np.random.uniform(np.deg2rad(-self.max_incl),np.deg2rad(self.max_incl)),
                        np.random.uniform(0, 2 * np.pi),
                    )
                    self.move_robot(pose)
                    image = self.image_storage.block(n_msgs=2)
                    image = stack_image(image, self.n_stackings)
                    if len(image.shape) == 2:
                        image = np.expand_dims(image, 0)
                    self.dataset.write_datapoint(input_data=image, label=pose)
        rospy.signal_shutdown("Dataset collected")
        self.ros_thread.join()


def main():
    node = DatasetCollectionNode()
    node.run()


if __name__ == "__main__":
    main()
