#!/usr/bin/env python3
# import rospy
import cv2
import numpy as np
from ObjectDetection import ObjectDetection
from DamageAnalysis import DamageAnalysis
from LogManager import LogManager


class TSDetections():
    # ===================================== INIT==========================================
    def __init__(self):
        # rospy.init_node('DetectionNode', anonymous=True)

        # YOLO object detection configurations
        yolo_model = '/home/can/desktop_thesis/results/yolov5/yolov5l/weights/best.pt' # rospy.get_param('/yolo_weight_file')
        yolo_path = '/home/can/desktop_thesis/yolov5' # rospy.get_param('/yolo_dir')
        model_size = 640 # rospy.get_param('/yolo_input_size')
        conf_thresh = 0.8 # rospy.get_param('/yolo_confidence')
        iou_thresh = 0.9 # rospy.get_param('/yolo_iou')
        self.OD = ObjectDetection(yolo_path, yolo_model, model_size, conf_thresh, iou_thresh)

        # Video input/output configurations
        video_save_dir = '/home/can/desktop_thesis/out_vid.avi' # rospy.get_param('/video_output_path')
        self.save_video = True # rospy.get_param('/save_video')
        self.debug_stream = True # rospy.get_param('/debug')
        self.save_output = False # rospy.get_param('/save_output')
        self.input_file = '/home/can/Desktop/uljana_dashcam/Normal/FILE221114-183908-000271.MOV' # rospy.get_param('/input_source')

        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.75
        self.font_color = (255, 0, 255)
        self.font_thickness = 1
        
        # Bounding box rectangle settings
        self.bbox_color = (0,0,255)
        self.bbox_thickness = 2

        # Autoencoder configurations for damage analysis
        self.sigma_multiplier = 2.0 # rospy.get_param('/sigma_multiplier')
        ae_weight_file = '/home/can/desktop_thesis/ae_weights/cropped_allfullmodel1mse.h5' # rospy.get_param('/autoencoder_model')
        self.comp_metric = 'ssim' # rospy.get_param('/comp_metric')
        iqa_file = '/home/can/damaged_ts_detection/src/ts_detection/iqa.yaml' # rospy.get_param('/iqa_file')
        self.DA = DamageAnalysis(iqa_file, self.comp_metric, ae_weight_file)

        # Logging settings
        self.log_root_dir = '/home/can/damaged_ts_detection/logs/' # rospy.get_param('/log_root_dir')
        self.lm = LogManager('/home/can/damaged_ts_detection/logs/')
        self.lm.write_meta(self.input_file, model_size, conf_thresh, iou_thresh, self.comp_metric, self.sigma_multiplier)

        # Opencv and frame settings
        self.cap = cv2.VideoCapture(self.input_file)
        frame_width = int(self.cap.get(3))
        frame_height = int(self.cap.get(4))
        vid_fps =(int(self.cap.get(cv2.CAP_PROP_FPS)))

        self.out_vid = cv2.VideoWriter(self.lm.folder_name+'/out_video.avi', 
                                       cv2.VideoWriter_fourcc('M','J','P','G'), 
                                       vid_fps, 
                                       (frame_width,frame_height))
        
        # Live detected sign display settings
        self.x_offset = 18
        self.y_offset = 18
        self.img_offset = 6
        self.orig_img_resize = 128
        self.gen_img_resize = 128
        self.frame_width = 1620
        self.frame_height = 1080
    

    def write_text(self, img, text, coords):
        cv2.putText(img, text, coords, self.font, self.font_scale, self.font_color, self.font_thickness, cv2.LINE_AA)


    def run_detection(self):
        frame_no = 0
        resized_crop = np.zeros((224, 224, 3), np.uint8)
        gen_img = np.zeros((224, 224, 3), np.uint8)
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            if ret == True:
                gen_imgs = []
                resized_crop_imgs = []
                class_ids, conf_vals, bboxes_coords, num_detections, cropped_imgs, yolo_elapsed_time = self.OD.detect_objects(frame, False)
                print("Num of detected ts: ", num_detections)

                for i in range(num_detections):
                    # YOLO detection outputs
                    class_id = class_ids[i]
                    crop_img = cropped_imgs[i]
                    conf_val = conf_vals[i]
                    if (self.save_video == True) or (self.debug_stream == True):
                        
                        xmin = bboxes_coords[i][0]
                        ymin = bboxes_coords[i][1]
                        xmax = bboxes_coords[i][2]
                        ymax = bboxes_coords[i][3]

                    # Damage analysis
                    dist_val, gen_img, ae_elapsed_time = self.DA.measure_distance(crop_img)
                    is_damaged, threshold = self.DA.check_damage(self.sigma_multiplier, class_id, dist_val)
                    ts_name = self.DA.ts_id_to_name(class_id)

                    resized_crop = cv2.resize(crop_img, (48,48), interpolation = cv2.INTER_AREA)
                    gen_imgs.append(gen_img)

                    # Display information on frames
                    if (self.save_video == True) or (self.debug_stream == True):
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), self.bbox_color, self.bbox_thickness)
                        self.write_text(frame, str(class_id) + ": " + ts_name, (xmin, ymin-60))
                        self.write_text(frame, "yolo conf: " + str(conf_val), (xmin, ymin-45))
                        self.write_text(frame, "SSIM Dist: " + str(dist_val), (xmin, ymin-30))
                        if is_damaged:
                            self.write_text(frame, "Damaged !!", (xmin, ymin-15))
                        else:
                            self.write_text(frame, "Not Damaged", (xmin, ymin-15))

                    resized_crop_imgs.append(resized_crop)

                    # Log the detected traffic sign
                    total_processing_time = yolo_elapsed_time + ae_elapsed_time
                    total_processing_time = round(total_processing_time, 3)
                    self.lm.log_frame_info(conf_val, ts_name, class_id, yolo_elapsed_time, 
                                           frame_no, self.comp_metric, dist_val, ae_elapsed_time, 
                                           threshold, is_damaged, total_processing_time)


                self.lm.save_images(resized_crop_imgs, gen_imgs, frame_no, num_detections)


                if (self.save_video == True):
                    resized_frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                    blank_image = np.zeros((1080, 1920, 3), np.uint8)
                    blank_image[0:self.frame_height, 300:1920] = resized_frame
                    for k in range(num_detections):
                        x_offset_orig = self.x_offset
                        x_offset_gen = self.x_offset + self.orig_img_resize + self.img_offset
                        y_offset = self.y_offset + (k * self.orig_img_resize)
                        resized_crop = cv2.resize(resized_crop_imgs[k], (self.orig_img_resize,self.orig_img_resize))
                        resized_gen = cv2.resize(gen_imgs[k], (self.gen_img_resize,self.gen_img_resize))
                        blank_image[y_offset:y_offset+resized_crop.shape[0], x_offset_orig:x_offset_orig+resized_crop.shape[1]] = resized_crop # orig
                        blank_image[y_offset:y_offset+resized_gen.shape[0], x_offset_gen:x_offset_gen+resized_gen.shape[1]] = resized_gen # gen

                    self.write_text(blank_image, 'Frame No: ' + str(frame_no), (0,960))
                    self.out_vid.write(blank_image)

                if (self.debug_stream == True):
                    cv2.imshow("output", blank_image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                frame_no += 1
            else:
                break

        print('Finished')
        self.cap.release()
        self.out_vid.release()

if __name__ == '__main__':
    # try:
    img_detect = TSDetections()
    img_detect.run_detection()
    # except rospy.ROSInterruptException:
    #     pass
    
    # rospy.spin()