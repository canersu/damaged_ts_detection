#!/usr/bin/env python3
from ultralytics import YOLO
from time import time
import numpy as np


class TSClassifier():
    # ===================================== INIT ==========================================
    def __init__(self, yolo_cls_model, input_cls_size, iou_cls_thresh):
        self.model_cls = YOLO(yolo_cls_model)  # load a pretrained model
        self.model_cls_size = input_cls_size
        self.iou_cls_thresh = iou_cls_thresh

    def classify_objects(self, img, conf_thresh):
        start_time = time()
        
        results = self.model_cls(img,
                                conf=conf_thresh, 
                                # half=True,
                                device=0,
                                iou=self.iou_cls_thresh,
                                imgsz=self.model_cls_size)
        
        
        boxes = results[0].boxes
        num_cls = len(results[0])
        cls_id = boxes.cls.to('cpu').numpy()
        cls_conf = boxes.conf.to('cpu').numpy()
        end_time = time()
        elapsed_time = end_time - start_time
        elapsed_time = round(elapsed_time, 3)
        
        return cls_id, cls_conf, elapsed_time, num_cls