#!/usr/bin/env python3
from ultralytics import YOLO
from time import time
import numpy as np


class ObjectDetection():
    # ===================================== INIT ==========================================
    def __init__(self, yolo_det_model, input_det_size, conf_det_thresh, iou_det_thresh):
        self.model_det = YOLO(yolo_det_model)  # load a pretrained model
        self.model_det_size = input_det_size
        self.conf_det_thresh = conf_det_thresh
        self.iou_det_thresh = iou_det_thresh

    # =============================== OBJECT DETECTION ====================================
    def detect_objects(self, img, normalized=False):
        start_time = time()
        cropped_imgs = []
        num_detections = 0
        
        results = self.model_det(img,
                                   conf=self.conf_det_thresh, 
                                   # half=True,
                                   device=0,
                                   iou=self.iou_det_thresh,
                                   imgsz=self.model_det_size)
        boxes = results[0].boxes
        num_detections = len(results[0])
        if normalized:
            bbox_coords = boxes.xyxyn.to('cpu').numpy()
        else:
            bbox_coords = boxes.xyxy.to('cpu').numpy()
        
        for i in range(num_detections):
            xmin = int(bbox_coords[i][0])
            ymin = int(bbox_coords[i][1])
            xmax = int(bbox_coords[i][2])
            ymax = int(bbox_coords[i][3])
            crop = img[ymin:ymax, xmin:xmax]
            cropped_imgs.append(crop)
        
        end_time = time()

        elapsed_time = end_time - start_time
        elapsed_time = round(elapsed_time, 3)
        
        return boxes.conf.to('cpu').numpy(), bbox_coords, num_detections, cropped_imgs, elapsed_time

