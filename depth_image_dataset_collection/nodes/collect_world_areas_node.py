#!/usr/bin/python
"""This node requires Gazebo to be already inited"""
import os
import rospy
import roslaunch
import time
from gazebo_msgs.srv import GetModelState, GetModelStateRequest, GetModelStateResponse
import numpy as np
import json
import threading


class AreaCollectorNode:
    def __init__(self):
        # Set class variables
        self.model_name = "unit_box_static"
        self.is_collecting = True
        self.current_pose = None
        # Init ros node, subscribers, publishers and services
        rospy.init_node("area_collection_node")
        self.folder_to_save_areas = rospy.get_param(
            "/area_collection/folder_of_current_world"
        )
        gazebo_get_model_state_service = "/gazebo/get_model_state"
        rospy.wait_for_service(gazebo_get_model_state_service)
        self.model_state_query_proxy = rospy.ServiceProxy(
            gazebo_get_model_state_service, GetModelState
        )
        # Init threads
        self.model_query_thread = threading.Thread(
            target=self.contiual_model_query_target
        )

    def spawn_unit_box(self):
        package = "gazebo_ros"
        executable = "spawn_model"
        path_to_model = "/home/lorenzo/model_editor_models/unit_box_static/model.sdf"
        node_args = f"-model {self.model_name} -file {path_to_model} -sdf"
        node = roslaunch.core.Node(package, executable, args=node_args)
        launch = roslaunch.scriptapi.ROSLaunch()
        launch.start()
        process = launch.launch(node)
        while process.is_alive():
            time.sleep(0.1)

    def query_unit_box_pose(self):
        response = self.model_state_query_proxy(
            GetModelStateRequest(self.model_name, "")
        )
        assert isinstance(response, GetModelStateResponse)
        x = response.pose.position.x
        y = response.pose.position.y
        z = response.pose.position.z
        return (x, y, z)

    def contiual_model_query_target(self):
        while self.is_collecting:
            self.current_pose = self.query_unit_box_pose()

    def run(self):
        self.spawn_unit_box()
        self.model_query_thread.start()
        areas = []
        while True:
            area_type = input(
                "Type of area to collect: [polygon, polyline] or [exit] or [save]"
            )
            if area_type.lower() in ["exit", "save"]:
                break
            elif area_type.lower() not in ["polygon", "polyline"]:
                print(f"The selected area '{area_type}' is not valid")
                continue
            else:
                print(f"Collecting area of type '{area_type}'")
            if area_type.lower() == "polygon":
                area = self.collect_polygon()
            elif area_type.lower() == "polyline":
                area = self.collect_polyline()
            inpt = input("Confirm area? [y/n]")
            if inpt.lower() == "y":
                areas.append(area)
        if area_type == "save":
            path_to_areas_file = os.path.join(self.folder_to_save_areas, "areas.json")
            with open(path_to_areas_file, "w+") as f:
                json.dump(areas, f)
        self.is_collecting = False
        self.model_query_thread.join()

    def collect_polygon(self):
        polygon = {"type": "polygon", "points": []}
        while True:
            inpt = input(
                "To add a point enter a, to delete last point enter b, to finish the poligon enter q: "
            )
            if inpt.lower() not in "abq":
                continue
            if inpt.lower() == "a":
                polygon["points"].append(self.current_pose)
            elif inpt.lower() == "b":
                polygon["points"].pop(-1)
            else:
                break
        return polygon

    def collect_polyline(self):
        polyline = {"type": "polyline", "points": [], "width": []}
        while True:  # Select Points
            inpt = input(
                "To add a point enter a, to delete last point enter b, to finish entering points enter q: "
            )
            if inpt.lower() not in "abq":
                continue
            if inpt.lower() == "a":
                polyline["points"].append(self.current_pose)
            elif inpt.lower() == "b":
                polyline["points"].pop(-1)
            elif inpt.lower() == "q":
                break
        while True:  # Select width
            inpt = input(
                "Entering width of the polyline, choose first point and press enter"
            )
            pose1 = self.current_pose
            inpt = input(
                "Entering width of the polyline, choose second point and press enter"
            )
            pose2 = self.current_pose
            inpt = input("If input is correct, select [a] else, select [b]")
            if inpt.lower() == "a":
                distance = np.linalg.norm(np.array(pose1) - np.array(pose2), ord=2)
                polyline["width"] = distance
                break
            else:
                continue
        return polyline


def main():
    node = AreaCollectorNode()
    node.run()


if __name__ == "__main__":
    main()
