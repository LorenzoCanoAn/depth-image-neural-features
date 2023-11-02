#!/bin/python
import rospy
import depth_image_neural_features.networks as nf
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch


class DepthImageNetworkTestingNode:
    def __init__(self):
        rospy.init_node("depth_image_testing_node")
        # Get params
        self.path_to_model = rospy.get_param("~path_to_model")
        self.image_input_topic = rospy.get_param("~image_input_topic")
        self.dist_est_output_topic = rospy.get_param(
            "~distance_estimation_output_topic"
        )
        # Init model
        self.model = nf.DeepDistEstimator()
        self.model.load_state_dict(torch.load(self.path_to_model))
        self.model = self.model.to('cuda')
        self.initial_feature = None
        # Bridge
        self.bridge = CvBridge()
        # Init pub and sub
        self.dist_estimation_publisher = rospy.Publisher(
            self.dist_est_output_topic,
            Float32,
        )
        rospy.Subscriber(self.image_input_topic, Image, self.callback)

    def callback(self, msg:Image):
        img = self.bridge.imgmsg_to_cv2(msg)
        img = torch.tensor(img).to("cuda")
        img = torch.reshape(img,(1,1,img.shape[0], img.shape[1]))
        if self.initial_feature is None:
            self.initial_feature = self.model.compress(img)
            return
        current_feature = self.model.compress(img) 
        distance = self.model.distance_funct(self.initial_feature, current_feature)
        print(distance.item())
    
    def run(self):
        rospy.spin()
        
        
def main():
    node = DepthImageNetworkTestingNode()
    node.run()
    
if __name__ == "__main__":
    main()
        