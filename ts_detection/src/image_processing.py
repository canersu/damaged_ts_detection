#!/usr/bin/env python3
from curses import raw
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


class ImProc():
    # ===================================== INIT==========================================
    def __init__(self):
        self.bridge = CvBridge()
        rospy.init_node('DetectionNode', anonymous=True)

        rospy.Subscriber('/thesis/ts_detection', detections, self.det_callback)
        rospy.Subscriber('/thesis/raw_image', Image, self.img_callback)

    def det_callback(self):
        pass

    def img_callback(self):
        pass
