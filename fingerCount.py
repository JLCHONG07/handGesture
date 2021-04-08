from cv2 import cv2
import time
import os
import handTrackMod as htm

widthCam, heightCam = 640,480

#folderPath="FingerImages"
#myList = os.listdir(folderPath)
#print(myList)
#overlayList =[]

#for imPath in myList:
    #image = cv2.imread(f'{folderPath}/{imPath}')
    #print(f'{folderPath}/{imPath}')
    #overlayList.append(image)

#print(len(overlayList))

pTime=0
cap=cv2.VideoCapture(0)
cap.set(3,widthCam)
cap.set(4,heightCam)

detector = htm.handDetector(detectionCon=0.75)
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img= cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    #print (lmList)

    if len(lmList) != 0:
        fingers = []

        #For Thumb
        if lmList[tipIds[0]][1] < lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)


        for id in range (1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
                
        #count total fingers
        totalFingers = fingers.count(1)
        print(totalFingers)

        cv2.putText(img,str(totalFingers)+" fingers",(45,375),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),2)

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img, f'FPS:{int(fps)}', (7, 70), cv2.FONT_HERSHEY_PLAIN, 3, (100, 255, 0), 3, cv2.LINE_AA)
    #cv2.putText(img,f'FPS:{int(fps)}',(50,50),
    #cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)


