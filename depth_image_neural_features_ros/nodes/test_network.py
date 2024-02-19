#!/home/lorenzo/miniconda3/envs/dinf/bin/python
import rospy
import depth_image_neural_features.networks as nf
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import torch
import numpy as np


def stack_image(image, n_stackings):
    height, width = image.shape
    assert width % 2**n_stackings == 0
    for n_stack in range(1, n_stackings + 1):
        new_width = int(width / 2**n_stack)
        image = np.vstack((image[:, :new_width], image[:, new_width:]))
    return image


class DepthImageNetworkTestingNode:
    def __init__(self):
        rospy.init_node("depth_image_testing_node")
        # Get params
        self.path_to_model = rospy.get_param("~path_to_model")
        self.image_input_topic = rospy.get_param("~image_input_topic")
        self.dist_est_output_topic = rospy.get_param("~distance_estimation_output_topic")
        # Init model
        self.model = nf.LastHope()
        self.model = self.model.to("cpu")
        self.model.eval()
        # self.model = torch.compile(self.model)
        self.model.load_state_dict(torch.load(self.path_to_model))
        self.model = self.model.to("cuda")
        self.initial_feature = None
        # Bridge
        self.bridge = CvBridge()
        # Init pub and sub
        self.dist_estimation_publisher = rospy.Publisher(
            self.dist_est_output_topic, Float32, queue_size=1
        )
        rospy.Subscriber(self.image_input_topic, Image, self.callback)
        self.x_publisher = rospy.Publisher("/x_diff", Float32, queue_size=1)
        self.y_publisher = rospy.Publisher("/y_diff", Float32, queue_size=1)
        self.angle_publisher = rospy.Publisher("/angle", Float32, queue_size=1)

    def callback(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg)
        # img = stack_image(img,3)
        img = torch.tensor(img).to("cuda")
        img = torch.reshape(img, (1, 1, img.shape[0], img.shape[1]))
        if self.initial_feature is None:
            self.initial_feature = self.model.compress(img)
            return
        current_feature = self.model.compress(img)
        distance = self.model.compare_features(current_feature, self.initial_feature)
        x = distance[0, 0].item()
        y = distance[0, 1].item()
        angle = np.arctan2(y, x)
        # z = distance[0,2].item()
        self.x_publisher.publish(Float32(x))
        self.y_publisher.publish(Float32(y))
        self.angle_publisher.publish(Float32(angle))
        # self.z_publisher.publish(Float32(z))

    def run(self):
        rospy.spin()


def main():
    node = DepthImageNetworkTestingNode()
    node.run()


if __name__ == "__main__":
    main()
