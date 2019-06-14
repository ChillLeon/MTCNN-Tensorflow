#coding:utf-8
import sys
import os
import tensorflow as tf
import numpy as np
import cv2
from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from train_models.mtcnn_model import P_Net, R_Net, O_Net
from prepare_data.loader import TestLoader

IMAGE_PATH = input("please input test image directory")
thresh = [0.6, 0.7, 0.7]
stride = 2
slide_window = False
detectors = [None, None, None]
shuffle = False
min_face_size = 20
prefix = ['../data/MTCNN_model/PNet_No_landmark/PNet', '../data/MTCNN_model/RNet_landmark/RNet', '../data/MTCNN_model/ONet_landmark/ONet']
epoch = [18, 14, 16]
batch_size = [2048, 64, 16]
model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]

#PNet model
if slide_window:
    PNet = Detector(P_Net, 12, model_path[0])
else:
    PNet = FcnDetector(P_Net, model_path[0])
detectors[0] = PNet

#RNet model
RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
detectors[1] = RNet

#ONet model
ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
detectors[2] = ONet

#get test images
test_img = []
img_path = "IMAGE_PATH"
for item in os.listdir(img_path):
    test_img.append(os.path.join(img_path, item))
test_img = TestLoader(test_img)

#draw boxes
mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size, stride=stride, 
    threshold=thresh, slide_window=slide_window)
boxes, landmarks = mtcnn_detector.detect_face(test_img)
count = 0
for imagePath in test_img:
    print(imagePath)
    image = cv2.imread(imagePath)
    for bbox in boxes[count]:
        cv2.putText(image, str(np.around(bbox[4], 2)), int(bbox[0]), int(bbox[1]), 
        cv2.FONT_HERSHEY_TRIPLEX, 1, color = (255, 0, 255))
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255))

    cv2.imshow("img%s",image) %count
    cv2.waitKey(0)
    count = count +1