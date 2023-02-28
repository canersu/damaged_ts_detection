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
import ts_detection.src.DamageAnalysis as DamageAnalysis


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
        self.save_video = rospy.get_param('/save_video')
        self.debug_stream = rospy.get_param('/debug')
        self.save_output = rospy.get_param('/save_output')

        # Opencv and frame settings
        self.cap = cv2.VideoCapture(self.input_file)
        self.out_vid = cv2.VideoWriter(video_save_dir, 
                                       cv2.VideoWriter_fourcc('M','J','P','G'), 
                                       vid_fps, 
                                       (frame_width,frame_height))
        
        frame_width = int(self.cap.get(3))
        frame_height = int(self.cap.get(4))
        vid_fps =(int(self.cap.get(cv2.CAP_PROP_FPS)))

        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.75
        self.font_color = (255, 0, 255)
        self.font_thickness = 1
        
        # Bounding box rectangle settings
        self.bbox_color = (0,0,255)
        self.bbox_thickness = 2

        # Autoencoder configurations for damage analysis
        self.sigma_multiplier = rospy.get_param('/sigma_multiplier')
        ae_weight_file = rospy.get_param('/autoencoder_model')
        comp_metric = rospy.get_param('/comp_metric')
        iqa_file = rospy.get_param('/iqa_file')
        self.DA = DamageAnalysis(iqa_file, comp_metric, ae_weight_file)


    def run_detection(self):
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            if ret == True:
                frame_yolo = frame[:, :, ::-1]
                class_ids, conf_vals, bboxes_coords, num_detections, cropped_imgs = self.OD.detect_objects(frame_yolo)

                for i in range(num_detections):
                    # YOLO detection outputs
                    class_id = class_ids[i]
                    crop_img = cropped_imgs[i]
                    if (self.save_video == True) or (self.debug_stream == True):
                        conf_val = conf_vals[i]
                        xmin = bboxes_coords[i][0]
                        ymin = bboxes_coords[i][1]
                        xmax = bboxes_coords[i][2]
                        ymax = bboxes_coords[i][3]

                    # Damage analysis
                    dist_val = self.DA.measure_distance(crop_img)
                    is_damaged = self.DA.check_damage(self.sigma_multiplier, class_id, dist_val)

                    # Display information on frames
                    if (self.save_video == True) or (self.debug_stream == True):
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), self.bbox_color, self.bbox_thickness)
                        cv2.putText(frame, str(class_id) + ": " + self.iqa_data[class_id]["name"], (xmin, ymin-60), font, fontScale, textcolor, textthickness, cv2.LINE_AA)
                        cv2.putText(frame, "yolo conf: " + str(conf), (xmin, ymin-45), font, fontScale, textcolor, textthickness, cv2.LINE_AA)