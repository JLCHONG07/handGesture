from cv2 import cv2
import time
import os
import hand.handTrackMod as htm

widthCam, heightCam = 640,480

#folderPath="FingerImages"
#myList = os.listdir(folderPath)
#print(myList)
#overlayList =[]

#for imPath in myList:
   # image = cv2.imread(f'{folderPath}/{imPath}')
    #print(f'{folderPath}/{imPath}')
   # overlayList.append(image)

#print(len(overlayList))
def rmStartMode1():
    cTime=0
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
                    
            if fingers==[0,1,1,0,0]:
                signMeaning="Peace"
            elif fingers==[1,1,0,0,1]:
                signMeaning = "Spider Man Skill"
            elif fingers==[1,1,1,1,1]:
                signMeaning="Hello!"
            elif fingers==[1,1,0,0,0]:
                signMeaning="Shoot pew pew"
            elif fingers==[0,1,1,1,0]:
                signMeaning="I swear"
            elif fingers==[0,0,1,1,1]:
                signMeaning="Ok"
            elif fingers==[0,0,1,0,0]:
                signMeaning="No! Don't do that!"
            elif fingers==[0,1,0,0,1]:
                signMeaning="You Rock!"
            elif fingers==[1,0,0,0,1]:
                signMeaning="666"
            elif fingers==[0,0,0,0,0]:
                signMeaning="Fist"
            else:
                signMeaning="No meaning..."



            #print(signMeaning)
            
            print(fingers)
            print(signMeaning)
            cv2.putText(img,str(signMeaning),(45,375),cv2.FONT_HERSHEY_PLAIN,4,(0,0,255),2)


        cTime= time.time()
        fps= 1/(cTime-pTime)
        pTime=cTime
        
        cv2.putText(img, f'FPS:{int(fps)}', (7, 70), cv2.FONT_HERSHEY_PLAIN, 3, (100, 255, 0), 3, cv2.LINE_AA)
        webFrame=  cv2.imencode('.jpg', img)[1].tobytes()
        yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + webFrame + b'\r\n'

        #cv2.imshow("Image", img)
        #cv2.waitKey(1)


