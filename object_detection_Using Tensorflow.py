import cv2
import tensorflow
import keras
import scipy

import imageai


net=cv2.dnn.readNet("dnn_model/yolov4-tiny.weights","dnn_model/yolov4-tiny.cfg")
model=cv2.dnn_DetectionModel(net)

model.setInputParams(size=(320,320),scale=1/255)
#loading all Classes which required
classes =[]

with open("dnn_model/classes.txt","r") as file_object:
    for class_name in file_object.readlines():
        print(class_name,end="")
        class_name=class_name.strip()
        classes.append(class_name)

print("Object List ")
print(classes)
# initilization camera
cap=cv2.VideoCapture(0)
# set size of detection
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
# for width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720);
# for height
# for FULL HD 1920 x 1080
# create Window
def click_button(event,x,y,flags,params):
    if event==cv2.EVENT_LBUTTONDOWN:
        print(x,y)

cv2.namedWindow("Frame ")
# cv2.setMouseCallback("Frame",click_button)
while True:
    try:
        ret, frame = cap.read()

        (class_ids,scores,bboxes)=model.detect(frame)

        for class_id,score,bbox in zip(class_ids,scores,bboxes):
            (x,y,w,h)=bbox
            class_name=classes[class_id]

            cv2.putText(frame,str(class_name),(x,y-5),cv2.FONT_HERSHEY_PLAIN,2,(200,0,50),2)

            cv2.rectangle(frame,(x,y),(x+w,y+h),(200,0,50),3)

        # print("class id's : ",class_ids)

        # print("Score : ",scores)

        # print("Boxes :",bboxes)

        cv2.imshow("Frame", frame)
        cv2.waitKey(1)
    except:
        print("Completed Succesfully")
        print("Cemera Stopped ")


