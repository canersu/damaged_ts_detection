#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import torch
import time
from ts_detection.msg import frame_info, detections, detection_info
from std_msgs.msg import Header
import ae
import tensorflow as tf
import yaml
import ts_detection.src.ObjectDetection as ObjectDetection


class TSDetections():
    # ===================================== INIT==========================================
    def __init__(self):
        rospy.init_node('DetectionNode', anonymous=True)

        # YOLO object detection configurations
        yolo_model = rospy.get_param('/yolo_weight_file')
        yolo_path = rospy.get_param('/yolo_dir')
        model_size = rospy.get_param('yolo_input_size')
        conf_thresh = rospy.get_param('yolo_confidence')
        iou_thresh = rospy.get_param('yolo_iou')
        self.OD = ObjectDetection(yolo_path, yolo_model, model_size, conf_thresh, iou_thresh)

        # Video input/output configurations
        video_save_dir = rospy.get_param('/video_output_path')
        self.save_video = rospy.get_param('/output_video')
        self.save_output = rospy.get_param('/save_output')


        # Opencv and frame settings
        self.cap = cv2.VideoCapture(self.input_file)
        self.out_vid = cv2.VideoWriter(video_save_dir, cv2.VideoWriter_fourcc('M','J','P','G'), vid_fps, (frame_width,frame_height))
        frame_width = int(self.cap.get(3))
        frame_height = int(self.cap.get(4))
        vid_fps =(int(self.cap.get(cv2.CAP_PROP_FPS)))
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 0.75
        self.textcolor = (255, 0, 255)
        self.textthickness = 1