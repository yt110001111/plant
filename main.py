import cv2
from object_detector import *
img = cv2.imread('pp.jpg') #the photo want to detect
#cap= cv2.VideoCapture(0)
#cap.set(3, 640)
#cap.set(4, 480)


classNames=[]
classFile='coco.names'
with open(classFile,'rt') as f:
    classNames=f.read().rstrip('\n').split('\n')

print(classNames)


configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

#while True:
    #success, img = cap.read()

classIDs, confs, bbox = net.detect(img, confThreshold=0.5)
i = classIDs
print(classIDs, bbox)

if not i is None:
    detector = HomogeneousBgDetector()
    contours = detector.detect_objects(img)
    print(contours)
    for cnt in contours:
        cv2.polylines(img, [cnt], True, (255, 0, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(0)


    #if len(classIDs) != 0:
for classID, confidence, box in zip(classIDs.flatten(), confs.flatten(), bbox):
        cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
        cv2.putText(img, classNames[classID-1].lower(), (box[0]+10, box[1]+30),
                cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
        cv2.putText(img, str(round(confidence*100, 2)), (box[0] + 500, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 300, 0), 2)


cv2.imshow("output", img)
cv2.waitKey(0)

