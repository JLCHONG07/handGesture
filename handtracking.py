from cv2 import cv2
import time
import os

widthCam, heightCam = 640,480

folderPath="FingerImages"
myList = os.listdir(folderPath)
print(myList)
overlayList =[]
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    #print(f'{folderPath}/{imPath}')
    overlayList.append(image)

print(len(overlayList))

pTime=0
cap=cv2.VideoCapture(0)
cap.set(3,widthCam)
cap.set(4,heightCam)

while True:
    success, img= cap.read()

    cTime= time.time()

    fps= 1/(cTime-pTime)
    pTime=cTime-0.01

    cv2.putText(img,f'FPS:{int(fps)}',(400,70),
    cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)


