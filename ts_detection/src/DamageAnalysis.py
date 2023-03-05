#!/usr/bin/env python3
import yaml
import cv2
import tensorflow as tf
import ae
from time import time

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
        start_time = time()
        crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        crop_rgb = crop_rgb/255.0
        resized_crop = cv2.resize(crop_rgb, (48,48), interpolation = cv2.INTER_AREA)
        resized_crop = resized_crop[None]
        gen = self.ae_model.predict(resized_crop)
        img_tensor = tf.convert_to_tensor(resized_crop, dtype=tf.float32)
        dist_val = self.ae_.compMetric(img_tensor, gen, self.comp_metric)
        dist_val = round(dist_val, 3)

        gen_img_np = gen.reshape(48,48,3)
        gen_img_np *= 255.0
        gen_img = cv2.cvtColor(gen_img_np, cv2.COLOR_RGB2BGR)

        end_time = time()
        elapsed_time = end_time - start_time
        elapsed_time = round(elapsed_time, 3)

        return dist_val, gen_img, elapsed_time


    # ================================= CHECK DAMAGE ======================================
    def check_damage(self, sigma_multiplier, class_id, distance):
        damaged = False
        mean = self.iqa_data[class_id][self.comp_metric]["mean"]
        sigma = self.iqa_data[class_id][self.comp_metric]["sigma"]
        threshold = mean + sigma_multiplier*sigma
        threshold = round(threshold, 3)
        
        if distance > threshold:
            damaged = True
        
        return damaged, threshold
    
    # ================================= TS ID TO NAME =====================================
    def ts_id_to_name(self, class_id):
        ts_name = self.iqa_data[class_id]["name"]
        return ts_name
