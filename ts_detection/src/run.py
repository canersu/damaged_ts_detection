#!/usr/bin/env python3
# import rospy
import cv2
import numpy as np
from ObjectDetection import ObjectDetection
from DamageAnalysis import DamageAnalysis
from LogManager import LogManager
from TSClassifier import TSClassifier


class TSDetections():
    # ===================================== INIT==========================================
    def __init__(self):
        # rospy.init_node('DetectionNode', anonymous=True)

        # YOLO object detection configurations
        det_yolo_model = '/home/can/storage/weight_files/damaged_ts/yolov8x_gtsdb_07_015_015/weights/best.pt' # rospy.get_param('/yolo_weight_file')
        # yolo_path = '/home/can/external_libraries/yolov5' # rospy.get_param('/yolo_dir')
        det_model_size = 1280 # rospy.get_param('/yolo_input_size')
        det_conf_thresh = 0.6 # rospy.get_param('/yolo_confidence')
        det_iou_thresh = 0.6 # rospy.get_param('/yolo_iou')
        self.OD = ObjectDetection(det_yolo_model, det_model_size, det_conf_thresh, det_iou_thresh)
        
        
        # YOLO object classification configurations
        cls_yolo_model = '/home/can/storage/weight_files/damaged_ts/yolov8x_gtsrb/weights/best.pt' # rospy.get_param('/yolo_weight_file')
        # yolo_path = '/home/can/external_libraries/yolov5' # rospy.get_param('/yolo_dir')
        cls_model_size = 64 # rospy.get_param('/yolo_input_size')
        self.cls_conf_thresh = 0.7 # rospy.get_param('/yolo_confidence')
        self.gen_cls_conf_thresh = 0.15 # AE gen threshold
        cls_iou_thresh = 0.7 # rospy.get_param('/yolo_iou')
        self.TSC = TSClassifier(cls_yolo_model, cls_model_size, cls_iou_thresh)

        # Video input/output configurations
        # video_save_dir = '/home/can/desktop_thesis/out_vid.avi' # rospy.get_param('/video_output_path')
        self.save_video = True # rospy.get_param('/save_video')
        self.debug_stream = True # rospy.get_param('/debug')
        self.save_output = False # rospy.get_param('/save_output')
        # self.input_file = '/home/can/storage/videos/damaged_ts/uljana_dashcam/Normal/FILE221114-183908-000271.MOV' # rospy.get_param('/input_source')
        self.input_file = '/home/can/storage/videos/damaged_ts/koblenz/koblenz_clip1.webm'
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1.25 # 0.75
        self.font_color = (255, 0, 255)
        self.font_thickness = 1
        
        # Bounding box rectangle settings
        self.bbox_color = (0,0,255)
        self.bbox_thickness = 2

        # Autoencoder configurations for damage analysis
        self.sigma_multiplier = 2.0 # rospy.get_param('/sigma_multiplier')
        ae_weight_file = '/home/can/storage/weight_files/damaged_ts/ae_weights/cropped_allfullmodel1mse.h5' # rospy.get_param('/autoencoder_model')
        self.comp_metric = 'ssim' # rospy.get_param('/comp_metric')
        iqa_file = '/home/can/damaged_ts_ws/damaged_ts_detection/ts_detection/iqa.yaml' # rospy.get_param('/iqa_file')
        self.DA = DamageAnalysis(iqa_file, self.comp_metric, ae_weight_file)

        # Logging settings
        self.log_root_dir = '/home/can/storage/logs/damaged_ts/' # rospy.get_param('/log_root_dir')
        self.lm = LogManager(self.log_root_dir)
        self.lm.write_meta(self.input_file, det_model_size, det_conf_thresh, self.gen_cls_conf_thresh, det_iou_thresh, cls_model_size, 
                           self.cls_conf_thresh, cls_iou_thresh, self.comp_metric, self.sigma_multiplier)

        # Opencv and frame settings
        self.cap = cv2.VideoCapture(self.input_file)
        frame_width = 1920 # int(self.cap.get(3))
        frame_height = 1080 # int(self.cap.get(4))
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
        # cv2.putText(img, text, coords, self.font, self.font_scale, self.font_color, self.font_thickness, cv2.LINE_AA)
        new_img = img
        new_img = cv2.putText(new_img, text, coords, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (0,0,0), self.font_thickness * 10)
        cv2.putText(new_img, text, coords, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255,255,255), self.font_thickness * 2)
        return new_img

    def run_detection(self):
        frame_no = 0
        resized_crop = np.zeros((224, 224, 3), np.uint8)
        gen_img = np.zeros((224, 224, 3), np.uint8)
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            if ret == True:
                x_new_frame = frame
                gen_imgs = []
                resized_crop_imgs = []
                cropped_imgs = []
                bboxes_coords = []
                conf_vals = []
                conf_cls_vals = []
                class_ids = []
                dist_vals = []
                verifieds = []
                gen_class_ids = []
                is_damageds = []
                num_detections = 0
                raw_conf_vals, raw_bboxes_coords, raw_num_detections, raw_cropped_imgs, yolo_elapsed_time = self.OD.detect_objects(x_new_frame, False)
                print("Num of raw detected ts: ", raw_num_detections)
                for i in range(raw_num_detections):
                    crop_img = raw_cropped_imgs[i]
                    raw_class_ids, cls_conf_val, cls_elapsed_time, num_cls = self.TSC.classify_objects(crop_img, self.cls_conf_thresh)
                    if num_cls !=0:
                        print("Detected class id: "+str(int(raw_class_ids[0])))
                        print("Detected class conf: " + str(float(cls_conf_val[0])))
                        class_id = int(raw_class_ids[0])
                        class_ids.append(class_id)
                        cropped_imgs.append(raw_cropped_imgs[i])
                        conf_vals.append(raw_conf_vals[i])
                        bboxes_coords.append(raw_bboxes_coords[i])
                        conf_cls_vals.append(cls_conf_val)
                        num_detections += 1
                        

                for i in range(num_detections):
                    # YOLO detection outputs
                    class_id = class_ids[i]
                    crop_img = cropped_imgs[i]
                    conf_val = conf_vals[i]
                    cls_conf_val = conf_cls_vals[i]
                    # if (self.save_video == True) or (self.debug_stream == True):
                        


                    # Damage analysis
                    dist_val, gen_img, ae_elapsed_time = self.DA.measure_distance(crop_img)
                    is_damaged, threshold = self.DA.check_damage(self.sigma_multiplier, class_id, dist_val)
                    ts_name = self.DA.ts_id_to_name(class_id)
                    dist_vals.append(dist_val)
                    
                    # Verify if the detected sign class and generated sign class are the same
                    gen_class_id, gen_cls_conf_val, cls_elapsed_time, gen_num_cls = self.TSC.classify_objects(gen_img,self.gen_cls_conf_thresh)
                    if gen_num_cls !=0:
                        print("Generated class id: " + str(int(gen_class_id[0])))
                        print("Generated class conf: " + str(float(gen_cls_conf_val[0])))
                        gen_class_id = int(gen_class_id[0])
                        gen_cls_conf_val = float(gen_cls_conf_val[0])
                    
                    log_verified = False
                    if gen_class_id == class_id:
                        log_verified = True

                    resized_crop = cv2.resize(crop_img, (48,48), interpolation = cv2.INTER_AREA)
                    gen_imgs.append(gen_img)
                    resized_crop_imgs.append(resized_crop)
                    
                    verifieds.append(log_verified)
                    gen_class_ids.append(gen_class_id)
                    is_damageds.append(is_damaged)


                    # Log the detected traffic sign
                    total_processing_time = yolo_elapsed_time + ae_elapsed_time
                    total_processing_time = round(total_processing_time, 3)
                    self.lm.log_frame_info(conf_val, cls_conf_val, gen_cls_conf_val, gen_class_id, log_verified, ts_name, class_id, yolo_elapsed_time, 
                                           frame_no, self.comp_metric, dist_val, ae_elapsed_time, 
                                           threshold, is_damaged, total_processing_time)



                self.lm.save_images(resized_crop_imgs, gen_imgs, frame_no, num_detections)
                
                for j in range(num_detections):
                    conf_val = conf_vals[j]
                    class_id = class_ids[j]
                    ts_name = self.DA.ts_id_to_name(class_id)
                    dist_val = dist_vals[j]
                    log_verified = verifieds[j]
                    dist_val = round(dist_val, 3)
                    conf_val = round(conf_val, 3)
                    gen_class_id = gen_class_ids[j]
                    gen_ts_name = "No Detection !"
                    if gen_class_id != None: 
                        gen_ts_name = self.DA.ts_id_to_name(gen_class_id)
                    is_damaged = is_damageds[j]
                    
                # Display information on frames
                    if (self.save_video == True) or (self.debug_stream == True):
                        xmin = int(bboxes_coords[j][0])
                        ymin = int(bboxes_coords[j][1])
                        xmax = int(bboxes_coords[j][2])
                        ymax = int(bboxes_coords[j][3])
                        # cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), self.bbox_color, self.bbox_thickness)
                        self.write_text(frame, str(class_id) + ": " + ts_name, (xmin, ymin-110))
                        self.write_text(frame, "csl conf: " + str(conf_val), (xmin, ymin-80))
                        self.write_text(frame, "SSIM Dist: " + str(dist_val), (xmin, ymin-45))
                        if log_verified:
                            self.write_text(frame, "Verified !", (xmin, ymin-15))
                        if not log_verified:
                            self.write_text(frame, "Gen ts cls: " + gen_ts_name, (xmin, ymin-15))
                        
                        if is_damaged:
                            # self.write_text(frame, "Damaged !!", (xmin, ymin-15))
                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), self.bbox_color, self.bbox_thickness*2)
                        else:
                            # self.write_text(frame, "Not Damaged", (xmin, ymin-15))
                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), self.bbox_thickness*2)

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
