#!/usr/bin/env python3
from curses import raw
from distutils.command.config import config
from chardet import detect
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import torch
import time
from ts_detection.msg import frame_info, detections
from std_msgs.msg import Header
import ae


class ImProc():
    # ===================================== INIT==========================================
    def __init__(self):
        self.bridge = CvBridge()
        self.f_info = frame_info()
        self.frame_list = detections()
        rospy.init_node('DetectionNode', anonymous=True)

        rospy.Subscriber('/thesis/ts_detection', detections, self.det_callback)
        rospy.Subscriber('/thesis/raw_image', Image, self.img_callback)

        self.crop_pub = rospy.Publisher('/thesis/cropped_ts',Image, queue_size=1)

    def det_callback(self, data):
        self.frame_list = data.frames

    def img_callback(self, msg):
        for f in self.frame_list:
            if len(self.frame_list) > 0:
                xmin = f.x1
                ymin = f.y1
                xmax = f.x2
                ymax = f.y2
                ts_id = f.class_id
                conf = f.confidence

                img = self.bridge.imgmsg_to_cv2(msg)

                crop = img[ymin:ymax, xmin:xmax]
                # print("x1: ",xmin,"  y1: ",ymin," x2: ",xmax," y2: ",ymax)
                print("TS ID: ", ts_id, " Confidence: ", conf)

                self.crop_pub.publish(self.bridge.cv2_to_imgmsg(crop, "bgr8"))



if __name__ == '__main__':
    try:
        improc  = ImProc()
    except rospy.ROSInterruptException:
        pass
    
    rospy.spin()