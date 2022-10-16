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
from ts_detection.msg import frame_info, detections
from std_msgs.msg import Header

class TSDetections():
    # ===================================== INIT==========================================
    def __init__(self):
        self.bridge = CvBridge()
        rospy.init_node('DetectionNode', anonymous=True)
        self.model = torch.hub.load('/home/can/thesis/yolov5', 'custom', path='/home/can/thesis/results/yolov5/yolov5l/weights/best.pt', source='local')
        self.model.conf = 0.75
        self.infile = "/home/can/thesis/notebooks/sample_video_01_cut.mp4"
        self.model_size = 640

        self.raw_image_pub = rospy.Publisher('/thesis/raw_image', Image, queue_size=1)
        self.detect_pub = rospy.Publisher('/thesis/ts_detection', detections, queue_size=1)

    def yolo_detections(self):
        cnt = 0
        cap = cv2.VideoCapture(self.infile)
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
                h = Header()
                h.stamp = rospy.Time.now()
                h.frame_id = 'bbox'
                # print('#Detected Signs: ', len(results.xyxy[0]), ' frame: ', cnt)
                for i in range(len(results.xyxy[0])):
                    image_info = frame_info()
                    xmin = int(results.xyxy[0][i][0])
                    ymin = int(results.xyxy[0][i][1])
                    xmax = int(results.xyxy[0][i][2])
                    ymax = int(results.xyxy[0][i][3])

                    #start_point = (xmin, ymin)
                    #end_point = (xmax, ymax)

                    conf = float(results.xyxy[0][i][4])
                    class_id = int(results.xyxy[0][i][5])
                    # print('#Detected Sign: ', class_id, ' frame: ', cnt)
                    image_info.x1 = xmin
                    image_info.x2 = ymin
                    image_info.y1 = xmax
                    image_info.y2 = ymax
                    image_info.class_id = class_id
                    image_info.confidence = conf

                    images.append(image_info)


                    #frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
                    #font = cv2.FONT_HERSHEY_SIMPLEX
                    #cv2.putText(frame, 'id' + class_id, (xmin,ymax), font, 0.8, (0,255,255), 2, cv2.LINE_AA)
                
                #cv2.imshow('Scene',frame)
                detects.frames = images
                detects.header = h
                self.detect_pub.publish(detects)

                raw_img = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                self.raw_image_pub.publish(raw_img)

                # if cv2.waitKey(25) & 0xFF == ord('q'):
                #    break

            else:
                break

        print('Total Frames: ' + str(cnt))
        cap.release()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Elapsed Time: ', elapsed_time)
        # cv2.destroyAllWindows()

        #final = self.bridge.cv2_to_imgmsg(self.final_img, "bgr8")

if __name__ == '__main__':
    try:
        img_detect = TSDetections()
        img_detect.yolo_detections()
    except rospy.ROSInterruptException:
        pass
    
    rospy.spin()