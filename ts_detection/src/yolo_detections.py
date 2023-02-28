#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import torch
import time
from ts_detection.msg import frame_info, detections, detection_info
from std_msgs.msg import Header
import ae
import tensorflow as tf
import yaml

class TSDetections():
    # ===================================== INIT==========================================
    def __init__(self):
        self.bridge = CvBridge()
        rospy.init_node('DetectionNode', anonymous=True)
        yolo_model = rospy.get_param('/yolo_weight_file')
        yolo_path = rospy.get_param('/yolo_dir')
        self.model = torch.hub.load(yolo_path, 'custom', path=yolo_model, source='local')
        self.input_file = rospy.get_param('/input_source')
        self.model_size = rospy.get_param('/yolo_input_size')
        self.model.conf = rospy.get_param('/yolo_confidence')
        self.model.iou = rospy.get_param('/yolo_iou')
        self.save_output = rospy.get_param('/save_output')
        self.debug = rospy.get_param('/debug')
        self.save_video = rospy.get_param('/output_video')

        self.det_img_out_dir = rospy.get_param('/detected_imgs_save_dir')

        self.raw_image_pub = rospy.Publisher('/thesis/raw_image', Image, queue_size=1)
        self.detect_pub = rospy.Publisher('/thesis/ts_detection', detections, queue_size=1)
        self.crop_pub = rospy.Publisher('/thesis/cropped_ts',Image, queue_size=1)
        self.det_pub = rospy.Publisher('/thesis/ae_det',detection_info, queue_size=1)

        self.ae_weight = rospy.get_param('/autoencoder_model')
        self.ae_ = ae.autoEncoder()
        self.ae_model = self.ae_.loadModel(self.ae_weight)

        # Load the YAML file into a Python dictionary
        with open("/home/can/damaged_ts_detection/src/ts_detection/iqa.yaml") as file:
            self.iqa_data = yaml.safe_load(file)
    
    def detect_objects(self, img, iou_thresh, conf_thresh, yolo_input_size):
        class_ids = []
        conf_vals = []
        bboxes_coords = []
        cropped_imgs = []

        img = img[:, :, ::-1]
        results = self.model(img, yolo_input_size)
        num_detections = len(results.xyxy[0])

        for i in range(num_detections):
            xmin = int(results.xyxy[0][i][0])
            ymin = int(results.xyxy[0][i][1])
            xmax = int(results.xyxy[0][i][2])
            ymax = int(results.xyxy[0][i][3])
            conf = float(results.xyxy[0][i][4])
            class_id = int(results.xyxy[0][i][5])

            bbox_coords = [xmin, ymin, xmax, ymax]
            crop = img[ymin:ymax, xmin:xmax]

            bboxes_coords.append(bbox_coords)
            class_ids.append(class_id)
            conf_vals.append(conf)
            cropped_imgs.append(crop)
        
        return class_ids, conf_vals, bboxes_coords, num_detections, cropped_imgs
    
    def distance_val(self, crop_img, comp_metric):
                 
        crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        crop_rgb = crop_rgb/255.0
        resized_crop = cv2.resize(crop_rgb, (48,48), interpolation = cv2.INTER_AREA)
        resized_crop = resized_crop[None]
        gen = self.ae_model.predict(resized_crop)
        img_tensor = tf.convert_to_tensor(resized_crop, dtype=tf.float32)
        val = self.ae_.compMetric(img_tensor, gen, comp_metric)
        return val



    def yolo_detections(self):
        cnt = 0
        det_cnt = 0
        xmin = 0
        xmax = 0
        ymin = 0
        ymax = 0
        cap = cv2.VideoCapture(self.input_file)
        start_time = time.time()
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        vid_fps =(int(cap.get(cv2.CAP_PROP_FPS)))
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.75
        textcolor = (255, 0, 255)
        textthickness = 1


        out_vid = cv2.VideoWriter("/home/can/damaged_ts_detection/denek_output.avi", cv2.VideoWriter_fourcc('M','J','P','G'), vid_fps, (frame_width,frame_height))
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                frame_yolo = frame[:, :, ::-1]
                cnt += 1

                results = self.model(frame_yolo, self.model_size)
                color = (0,0,255)
                thickness = 2
                images = []
                detects = detections()
                image_info = frame_info()
                det_info = detection_info()
                h = Header()
                h.stamp = rospy.Time.now()
                h.frame_id = 'bbox'
                for i in range(len(results.xyxy[0])):
                    image_info = frame_info()
                    xmin = int(results.xyxy[0][i][0])
                    ymin = int(results.xyxy[0][i][1])
                    xmax = int(results.xyxy[0][i][2])
                    ymax = int(results.xyxy[0][i][3])

                    conf = float(results.xyxy[0][i][4])
                    class_id = int(results.xyxy[0][i][5])
                    # print('#Detected Sign: ', class_id, ' frame: ', cnt)
                    image_info.x1 = xmin
                    image_info.y1 = ymin
                    image_info.x2 = xmax
                    image_info.y2 = ymax
                    image_info.class_id = class_id
                    image_info.confidence = conf

                    images.append(image_info)

                    if (self.save_video==True) or (self.debug==True):
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness)
                        cv2.putText(frame, str(class_id) + ": " + self.iqa_data[class_id]["name"], (xmin, ymin-60), font, fontScale, textcolor, textthickness, cv2.LINE_AA)
                        cv2.putText(frame, "yolo conf: " + str(conf), (xmin, ymin-45), font, fontScale, textcolor, textthickness, cv2.LINE_AA)

                        crop = frame[ymin:ymax, xmin:xmax]
                        print("TS ID: ", class_id, " Confidence: ", conf)
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        crop_rgb = crop_rgb/255.0
                        resized_crop = cv2.resize(crop_rgb, (48,48), interpolation = cv2.INTER_AREA)
                        resized_crop = resized_crop[None]
                        gen = self.ae_model.predict(resized_crop)
                        img_tensor = tf.convert_to_tensor(resized_crop, dtype=tf.float32)
                        val = self.ae_.compMetric(img_tensor, gen, "SSIM")
                        cv2.putText(frame, "SSIM Dist: " + str(val), (xmin, ymin-30), font, fontScale, textcolor, textthickness, cv2.LINE_AA)
                        threshold = self.iqa_data[class_id]["ssim"]["mean"] + 2.0*self.iqa_data[class_id]["ssim"]["sigma"]
                        if val >= threshold:
                            cv2.putText(frame, "Damaged !!", (xmin, ymin-15), font, fontScale, textcolor, textthickness, cv2.LINE_AA)
                        else:
                            cv2.putText(frame, "Not Damaged", (xmin, ymin-15), font, fontScale, textcolor, textthickness, cv2.LINE_AA)

                detects.frames = images
                detects.header = h
                self.detect_pub.publish(detects)

                if self.save_video:
                    out_vid.write(frame)
                    cv2.imshow("output", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if self.debug:
                    raw_img = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                    self.raw_image_pub.publish(raw_img)
                # if len(images) > 0:
                #     crop = frame[ymin:ymax, xmin:xmax]
                #     if self.debug:
                #         self.crop_pub.publish(self.bridge.cv2_to_imgmsg(crop, "bgr8"))
                #     print("TS ID: ", class_id, " Confidence: ", conf)
                #     if self.save_output:
                #         cv2.imwrite(self.det_img_out_dir+str(class_id)+'/'+str(det_cnt)+'.png',crop)
                #     crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                #     crop_rgb = crop_rgb/255.0
                #     resized_crop = cv2.resize(crop_rgb, (48,48), interpolation = cv2.INTER_AREA)
                #     resized_crop = resized_crop[None]

                #     gen = self.ae_model.predict(resized_crop)
                #     img_tensor = tf.convert_to_tensor(resized_crop, dtype=tf.float32)
                #     val = self.ae_.compMetric(img_tensor, gen, "SSIM")
                #     det_info.confidence = conf
                #     det_info.class_id = class_id
                #     det_info.ssim_comp = val
                #     det_info.id = det_cnt
                #     self.det_pub.publish(det_info)
                #     det_cnt += 1
            else:
                break

        print('Total Frames: ' + str(cnt))
        cap.release()
        out_vid.release()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Elapsed Time: ', elapsed_time)

if __name__ == '__main__':
    try:
        img_detect = TSDetections()
        time.sleep(2)
        img_detect.yolo_detections()
    except rospy.ROSInterruptException:
        pass
    
    rospy.spin()