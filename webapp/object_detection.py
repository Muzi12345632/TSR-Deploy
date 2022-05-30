import cv2

##TODO download weights file  from opencv frozen_inference_graph
##TODO ssd mobilenet_v3_large_coco

cap= cv2.VideoCapture(0)

classNames=[]
classfile= 'traffic_sign_classes'

with open(classfile, 'rt') as f:
    classNames= f.read().rstrip('\n').split('\n')
    print(classNames)

configpath=
weightspath=


net= cv2.dnn_DetectionModel()
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5,127.5))
net.setInputSwapRB(True)


while True:
    success, img = cap.read()

    classIds, confs, bbox= net.detect(img, confThreshold=0.5)
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(0,255,0),thickness=2)


    cv2.imshow('result', img)
    cv2.waitKey(1)

