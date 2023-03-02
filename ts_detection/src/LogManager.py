#!/usr/bin/env python3
import os
from datetime import datetime

class LogManager():
    def __init__(self, root_dir):
        # Get the current date and time
        now = datetime.now()
        # Create a folder with the current date and time as its name
        self.folder_name = root_dir + now.strftime('%Y-%m-%d_%H-%M-%S')
        os.mkdir(self.folder_name)

        # Create detected_imgs folder
        os.mkdir(self.folder_name+'/detected_imgs')

        # Create generated_imgs folder
        os.mkdir(self.folder_name+'/generated_imgs')

        # Create the log file
        self.log_file = self.folder_name + '/logs.csv'
        with open(self.log_file, 'w') as f:
            pass

        # Create meta file
        self.meta_file = self.folder_name + '/Meta.txt'
        with open(self.meta_file, 'w') as f:
            pass


    def write_meta(self, input_vid_name, yolo_input_size, yolo_conf_thresh, 
                   yolo_iou_thresh, comp_metric, sigma_multiplier):
        rows = [
            ['Video name: ' + input_vid_name],
            ['YOLO input size: ' + yolo_input_size],
            ['YOLO confidence threshold: ' + yolo_conf_thresh],
            ['YOLO IOU threshold: ' + yolo_iou_thresh],
            ['Comparison metric: ' + comp_metric],
            ['Sigma multiplier: ' + sigma_multiplier]
        ]
        with open(self.meta_file, 'a') as f:
            for row in rows:
                line = ','.join(row) + '\n'
                f.write(line)
            

    def save_images(self, roi_img, gen_img, frame_no, cnt):
        pass

 
    def log_frame_info(self):
        pass

if __name__ == '__main__':
    lm = LogManager('/home/can/damaged_ts_detection/logs/')
