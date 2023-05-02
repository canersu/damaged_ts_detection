#!/usr/bin/env python3
import os
from datetime import datetime
import cv2
import csv

class LogManager():
    def __init__(self, root_dir):
        # Get the current date and time
        now = datetime.now()
        # Create a folder with the current date and time as its name
        self.folder_name = root_dir + now.strftime('%Y-%m-%d_%H-%M-%S')
        os.mkdir(self.folder_name)

        # Create detected_imgs folder
        self.det_img_dir = self.folder_name+'/detected_imgs/'
        os.mkdir(self.det_img_dir)

        # Create generated_imgs folder
        self.gen_img_dir = self.folder_name+'/generated_imgs/'
        os.mkdir(self.gen_img_dir)

        # Create the log file
        self.columns = ['frame_no', 'detected_class_id', 'detected_class', 'confidence_det',
                   'confidence_cls', 'comp_metric', 'threshold_value', 'comp_distance', 'is_damaged',
                   'yolo_detection_time', 'damage_analysis_time', 'total_processing_time']
        
        self.log_file = self.folder_name + '/logs.csv'
        with open(self.log_file , 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.columns)
            writer.writeheader()

        # Create meta file
        self.meta_file = self.folder_name + '/Meta.txt'
        with open(self.meta_file, 'w') as f:
            pass


    def write_meta(self, input_vid_name, yolo_input_size_det,
                   yolo_conf_thresh_det, yolo_iou_thresh_det,
                   yolo_input_size_cls, yolo_conf_thresh_cls,
                   yolo_iou_thresh_cls, comp_metric, sigma_multiplier):
        rows = [
            ['Video name: ' + input_vid_name],
            ['YOLO detection input size: ' + str(yolo_input_size_det)],
            ['YOLO classification input size: ' + str(yolo_input_size_cls)],
            ['YOLO detection confidence threshold: ' + str(yolo_conf_thresh_det)],
            ['YOLO classification confidence threshold: ' + str(yolo_conf_thresh_cls)],
            ['YOLO detection IOU threshold: ' + str(yolo_iou_thresh_det)],
            ['YOLO classification IOU threshold: ' + str(yolo_iou_thresh_cls)],
            ['Comparison metric: ' + comp_metric],
            ['Sigma multiplier: ' + str(sigma_multiplier)]
        ]
        with open(self.meta_file, 'a') as f:
            for row in rows:
                line = ','.join(row) + '\n'
                f.write(line)
            

    def save_images(self, roi_imgs, gen_imgs, frame_no, num_detected_imgs):
        for i in range(num_detected_imgs):
            cv2.imwrite(self.det_img_dir+str(frame_no)+'_'+str(i)+'.png', roi_imgs[i])
            cv2.imwrite(self.gen_img_dir+str(frame_no)+'_'+str(i)+'.png', gen_imgs[i])

 
    def log_frame_info(self, confidence_det, confidence_cls, detected_class, detected_class_id, yolo_detection_time, 
                       frame_no, comp_metric, comp_distance, damage_analysis_time, 
                       threshold_value, is_damaged, total_processing_time):
        with open(self.log_file , 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.columns)
            writer.writerow({'frame_no': str(frame_no), 'detected_class_id': str(detected_class_id), 
                                'detected_class': detected_class, 'confidence_det': str(confidence_det),
                                'confidence_cls': str(confidence_cls),
                                'comp_metric': comp_metric, 'threshold_value': str(threshold_value), 
                                'comp_distance': str(comp_distance), 'is_damaged': str(is_damaged), 
                                'yolo_detection_time': str(yolo_detection_time), 
                                'damage_analysis_time': str(damage_analysis_time), 
                                'total_processing_time': str(total_processing_time)})
        
        # columns = ['frame_no', 'detected_class_id', 'detected_class', 'confidence',
        #            'comp_metric', 'threshold_value', 'comp_distance', 'is_damaged',
        #            'yolo_detection_time', 'damage_analysis_time', 'total_processing_time']
        
        # if not os.path.exists(self.log_file):
        #     with open(self.log_file , 'w', newline='') as csvfile:
        #         writer = csv.DictWriter(csvfile, fieldnames=columns)
        #         writer.writeheader()
        # else:
        #     print(f"{filename} already exists, so it was not created.")



# if __name__ == '__main__':
#     lm = LogManager('/home/can/damaged_ts_detection/logs/')
