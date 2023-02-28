#!/usr/bin/env python3
import torch
import yaml
import cv2
import tensorflow as tf
import ae


class DamageAnalysis():
    # ===================================== INIT ==========================================
    def __init__(self, iqa_file, comp_metric, ae_weight_file):
        with open(iqa_file) as file:
            self.iqa_data = yaml.safe_load(file)
        
        self.comp_metric = comp_metric
        self.ae_weight = ae_weight_file
        self.ae_ = ae.autoEncoder()
        self.ae_model = self.ae_.loadModel(self.ae_weight)

    
    # =============================== MEASURE DISTANCE ====================================
    def measure_distance(self, crop_img):
        crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        crop_rgb = crop_rgb/255.0
        resized_crop = cv2.resize(crop_rgb, (48,48), interpolation = cv2.INTER_AREA)
        resized_crop = resized_crop[None]
        gen = self.ae_model.predict(resized_crop)
        img_tensor = tf.convert_to_tensor(resized_crop, dtype=tf.float32)
        dist_val = self.ae_.compMetric(img_tensor, gen, self.comp_metric)

        return dist_val


    # ================================= CHECK DAMAGE ======================================
    def check_damage(self, sigma_multiplier, class_id, distance):
        damaged = False
        mean = self.iqa_data[class_id][self.comp_metric]["mean"]
        sigma = self.iqa_data[class_id][self.comp_metric]["sigma"]
        threshold = mean + sigma_multiplier*sigma
        
        if distance > threshold:
            damaged = True
        
        return damaged