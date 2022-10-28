#!/usr/bin/env python3
from curses import raw
from chardet import detect
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

class TSDetections():
    # ===================================== INIT==========================================
    def __init__(self):
        self.bridge = CvBridge()
        rospy.init_node('DetectionNode', anonymous=True)
        yolo_model = rospy.get_param('/yolo_weight_file')
        yolo_path = rospy.get_param('/yolo_dir')
        self.model = torch.hub.load(yolo_path, 'custom', path=yolo_model, source='local')
        self.input_file = rospy.get_param('/input_source')
        self.model_size = rospy.get_param('yolo_input_size')
        self.model.conf = rospy.get_param('yolo_confidence')
        self.model.iou = rospy.get_param('yolo_iou')
        self.det_img_out_dir = "/home/can/thesis/ros_detections/"

        self.raw_image_pub = rospy.Publisher('/thesis/raw_image', Image, queue_size=1)
        self.detect_pub = rospy.Publisher('/thesis/ts_detection', detections, queue_size=1)
        self.crop_pub = rospy.Publisher('/thesis/cropped_ts',Image, queue_size=1)
        self.det_pub = rospy.Publisher('/thesis/ae_det',detection_info, queue_size=1)

        self.ae_weight = "/home/can/thesis/ae_weights/cropped_allfullmodel1mse.h5"
        self.ae_ = ae.autoEncoder()
        self.ae_model = self.ae_.loadModel(self.ae_weight)

    def yolo_detections(self):
        cnt = 0
        det_cnt = 0
        xmin = 0
        xmax = 0
        ymin = 0
        ymax = 0
        cap = cv2.VideoCapture(self.input_file)
        start_time = time.time()
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                frame_yolo = frame[:, :, ::-1]
                cnt += 1

                results = self.model(frame_yolo, self.model_size)
                #color = (0,0,255)
                #thickness = 2
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

                detects.frames = images
                detects.header = h
                self.detect_pub.publish(detects)

                raw_img = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                self.raw_image_pub.publish(raw_img)
                if len(images) > 0:
                    crop = frame[ymin:ymax, xmin:xmax]
                    self.crop_pub.publish(self.bridge.cv2_to_imgmsg(crop, "bgr8"))
                    print("TS ID: ", class_id, " Confidence: ", conf)
                    cv2.imwrite(self.det_img_out_dir+str(class_id)+'/'+str(det_cnt)+'.png',crop)
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    crop_rgb = crop_rgb/255.0
                    resized_crop = cv2.resize(crop_rgb, (48,48), interpolation = cv2.INTER_AREA)
                    resized_crop = resized_crop[None]

                    gen = self.ae_model.predict(resized_crop)
                    img_tensor = tf.convert_to_tensor(resized_crop, dtype=tf.float32)
                    val = self.ae_.compMetric(img_tensor, gen, "SSIM")
                    det_info.confidence = conf
                    det_info.class_id = class_id
                    det_info.ssim_comp = val
                    det_info.id = det_cnt
                    self.det_pub.publish(det_info)
                    det_cnt += 1
            else:
                break

        print('Total Frames: ' + str(cnt))
        cap.release()
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