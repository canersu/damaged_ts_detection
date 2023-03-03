#!/usr/bin/env python3
import torch
from time import time


class ObjectDetection():
    # ===================================== INIT ==========================================
    def __init__(self, yolo_path, yolo_model, input_size, conf_thresh, iou_thresh):
        self.model = torch.hub.load(yolo_path, 'custom', path=yolo_model, source='local')
        self.model_size = input_size
        self.model.conf = conf_thresh
        self.model.iou = iou_thresh

    # =============================== OBJECT DETECTION ====================================
    def detect_objects(self, img, normalized=False):
        start_time = time()
        class_ids = []
        conf_vals = []
        bboxes_coords = []
        cropped_imgs = []
        img_height = float(img.shape[0])
        img_width = float(img.shape[1])

        img_yolo = img[:, :, ::-1]
        results = self.model(img_yolo, self.model_size)
        num_detections = len(results.xyxy[0])

        for i in range(num_detections):
            xmin = int(results.xyxy[0][i][0])
            ymin = int(results.xyxy[0][i][1])
            xmax = int(results.xyxy[0][i][2])
            ymax = int(results.xyxy[0][i][3])
            conf = float(results.xyxy[0][i][4])
            class_id = int(results.xyxy[0][i][5])

            if normalized:
                bbox_coords = [xmin/img_width, ymin/img_height, 
                               xmax/img_width, ymax/img_height]
            else:
                bbox_coords = [xmin, ymin, xmax, ymax]
            
            crop = img[ymin:ymax, xmin:xmax]

            bboxes_coords.append(bbox_coords)
            class_ids.append(class_id)
            conf_vals.append(conf)
            cropped_imgs.append(crop)
        
        end_time = time()

        elapsed_time = end_time - start_time
        elapsed_time = round(elapsed_time, 3)
        
        return class_ids, conf_vals, bboxes_coords, num_detections, cropped_imgs, elapsed_time
