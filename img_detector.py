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

#IMAGE_PATH = input("please input test image directory")
thresh = [0.3, 0.4, 0.4]
stride = 2
slide_window = False
detectors = [None, None, None]
shuffle = False
min_face_size = 20
#prefix = ['../data/MTCNN_model/PNet_landmark/PNet', '../data/MTCNN_model/RNet_landmark/RNet', '../data/MTCNN_model/ONet_landmark/ONet']
prefix = ['E:\\facesample_neg\\MTCNN-Tensorflow-master\\data\\MTCNN_model\\PNet_landmark', 
    'E:\\facesample_neg\\MTCNN-Tensorflow-master\\data\\MTCNN_model\\RNet_landmark\\RNet-14',
    'E:\\facesample_neg\\MTCNN-Tensorflow-master\\data\\MTCNN_model\\ONet_landmark\\ONet-16']    
epoch = [18, 14, 16]
batch_size = [2048, 64, 16]
#model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
model_path = ['%s' %x for x in prefix]
print ('model_path vector is: ',model_path)

#PNet model
if slide_window:
    PNet = Detector(P_Net, 12, batch_size[0],model_path[0])
else:
    PNet = FcnDetector(P_Net, model_path[0])
detectors[0] = PNet

#RNet model
RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
detectors[1] = RNet

#ONet model
ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
detectors[2] = ONet

mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size, stride=stride, 
    threshold=thresh, slide_window=slide_window)
#get test images
test_img = []
img_path = "E:\\facesample_neg\\test_image"
for item in os.listdir(img_path):
    test_img.append(os.path.join(img_path, item))
img_data = TestLoader(test_img)

#draw boxes
boxes, landmarks = mtcnn_detector.detect_face(img_data)
count = 0
min_w = 441
min_h = 358
print(count)
for imagePath in test_img:
    image = cv2.imread(imagePath)
    #print("here")
    for bbox in boxes[count]:
        #cv2.putText(image, str(np.around(bbox[4], 2)), int(bbox[0]), int(bbox[1]), cv2.FONT_HERSHEY_TRIPLEX, 1, color = (255, 0, 255))
        image_new = image[int(bbox[1]-30):int(bbox[1]+328),int(bbox[0]-40):int(bbox[0])+401]
        #img_mark = cv2.rectangle(image, (int(bbox[0]),int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 0))
        img_resized = cv2.resize(image_new, (358, 441), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('E:\\facesample_neg\\test_img_output\\test_image_%d.jpg'%(count),img_resized) 
        #cv2.imshow("mark_img",img_mark)
        #cv2.imshow("img",image_new)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    count = count +1
print("total image number is %d"%(count))