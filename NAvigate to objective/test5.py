#!/usr/bin/env python3
import torch
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import rospy
import time
import numpy as np
from interbotix_xs_modules.locobot import InterbotixLocobotXS

class PhoneFinder:
    def __init__(self):
        # Initialize robot (InterbotixLocobotXS already calls rospy.init_node internally)
        self.robot = InterbotixLocobotXS(robot_model="locobot_px100")
        self.bridge = CvBridge()

        # Publisher for base velocity
        self.cmd_pub = rospy.Publisher("/mobile_base/cmd_vel", Twist, queue_size=10)

        # YOLOv5 model (pretrained)
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.phone_detected = False
        self.cv_image = None

        # Subscribe to camera topic
        rospy.Subscriber("/locobot/camera/color/image_raw", Image, self.image_callback)
        self.rate = rospy.Rate(10)

        


    def image_callback(self, msg):
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def scan_camera(self):
        tilt_angles = np.linspace(-1, 1, 5)        # tilt down -> up
        pan_angles  = np.linspace(-1.57, 1.57, 7)  # pan left -> right

        print("Scanning for phone...")
        while not self.phone_detected and not rospy.is_shutdown():
            for tilt in tilt_angles:
                self.robot.camera.tilt(tilt)
                for pan in pan_angles:
                    self.robot.camera.pan(pan)
                    rospy.sleep(0.5)
                    if self.cv_image is None:
                        continue
                    results = self.model(self.cv_image)
                    self.visualize_detections(results)
                    for *box, conf, cls in results.xyxy[0]:
                        label = int(cls)
                        if self.model.names[label] == "backpack":
                            self.phone_detected = True
                            x1, y1, x2, y2 = box
                            self.phone_center = ((x1+x2)/2, (y1+y2)/2)
                            print("Phone detected!")
                            return

    def approach_phone(self):
        print("Approaching phone...")
        while not rospy.is_shutdown():
            if self.cv_image is None:
                continue
            results = self.model(self.cv_image)
            phone_found = False
            for *box, conf, cls in results.xyxy[0]:
                label = int(cls)
                if self.model.names[label] == "cell phone":
                    phone_found = True
                    x1, y1, x2, y2 = box
                    cx = (x1+x2)/2
                    error_x = cx - self.cv_image.shape[1]/2
                    vel = Twist()
                    vel.linear.x = 0.2
                    vel.angular.z = -0.002 * error_x
                    self.cmd_pub.publish(vel)
                    print("Found found at",vel)
                    break
            if not phone_found:
                self.cmd_pub.publish(Twist())  # stop if lost
                break
            self.rate.sleep()
        self.cmd_pub.publish(Twist())  # stop
        
    def pickup(self):
        self.arm.go_to_home_pose()
        self.arm.set_single_joint_position("waist", 0)
        self.arm.set_ee_cartesian_trajectory(x=-0.03, z= -0.21)
        self.arm.set_ee_cartesian_trajectory(x=0.02)
        self.gripper.close(2.0)
        self.arm.set_ee_cartesian_trajectory(z=0.1)
	
    def drop(self):
        self.arm.set_ee_cartesian_trajectory(z=-0.1)
        self.gripper.open(2.0)
        self.arm.set_ee_cartesian_trajectory(x=-0.02)
        self.arm.go_to_sleep_pose()

    def run(self):
        self.scan_camera()
        if self.phone_detected:
            #self.approach_phone()
            print("Arrived near phone!")
        else:
            print("Phone not found.")
    def visualize_detections(self, results):
        if self.cv_image is None:
            return
        img = self.cv_image.copy()
        for *box, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, box)
            label = f"{self.model.names[int(cls)]} {conf:.2f}"

            # Draw box and label
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2
            )   

        cv2.imshow("YOLOv5 Debug View", img)
        cv2.waitKey(1)
if __name__ == "__main__":
    finder = PhoneFinder()
    finder.run()