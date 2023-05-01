#!/usr/bin/env python3
from ultralytics import YOLO
from time import time
import numpy as np


class TSClassifier():
    # ===================================== INIT ==========================================
    def __init__(self, yolo_cls_model, input_cls_size, conf_cls_thresh, iou_cls_thresh):
        self.model_cls = YOLO(yolo_cls_model)  # load a pretrained model
        self.model_cls_size = input_cls_size
        self.conf_cls_thresh = conf_cls_thresh
        self.iou_cls_thresh = iou_cls_thresh

    def classify_objects(self, img, normalized=False):
        start_time = time()
        
        results = self.model(img,
                                conf=self.conf_thresh, 
                                # half=True,
                                device=0,
                                iou=self.iou_thresh,
                                imgsz=self.model_size)
        
        
        end_time = time()

        elapsed_time = end_time - start_time
        elapsed_time = round(elapsed_time, 3)
        
        return boxes.cls.to('cpu').numpy(), boxes.conf.to('cpu').numpy(), elapsed_time