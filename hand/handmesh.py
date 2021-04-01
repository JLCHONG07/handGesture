import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def openCam():
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        min_detection_confidence=0.75,
        max_num_hands=2,
        min_tracking_confidence=0.75)as hands:
    
     while cap.isOpened():
        success, image = cap.read()
        if not success:
         print("Ignoring empty camera frame.")
         # If loading a video, use 'break' instead of 'continue'.
         continue
        else:
         #build frame/cam on web
         #frame = cv2.imencode('.jpg', image)[1].tobytes()
         frame=madeImage(image,hands)
         #print("debug"+frame.toString())
         webFrame = cv2.imencode('.jpg', frame)[1].tobytes()
         yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + webFrame + b'\r\n'

def madeImage(image,hands):
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)
        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
              mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        #cv2.imshow('MediaPipe Hands', image)
        return image
        #self.frameOnWeb(image)    
        #if cv2.waitKey(5) & 0xFF == 27:
            #break

        #build frame/cam on web
        #frame = cv2.imencode('.jpg', image)[1].tobytes()
        #yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'

def secondMode():
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        min_detection_confidence=0.75,
        max_num_hands=2,
        min_tracking_confidence=0.75)as hands:
    
     while cap.isOpened():
        success, image = cap.read()
        if not success:
         print("Ignoring empty camera frame.")
         # If loading a video, use 'break' instead of 'continue'.
         continue
        else:
         #build frame/cam on web
         #frame = cv2.imencode('.jpg', image)[1].tobytes()
         frame=secondModeImage(image,hands)
         #print("debug"+frame.toString())
         webFrame = cv2.imencode('.jpg', frame)[1].tobytes()
         yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + webFrame + b'\r\n'

def secondModeImage(image,hands):
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)
        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #make it show only joint of hands
        mask=np.zeros(image.shape[:],dtype="uint8")
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
              mp_drawing.draw_landmarks(
                mask, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        #cv2.imshow('MediaPipe Hands', image)
        return mask
        #self.frameOnWeb(image)    
        #if cv2.waitKey(5) & 0xFF == 27:
            #break

        #build frame/cam on web
        #frame = cv2.imencode('.jpg', image)[1].tobytes()
        #yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'

     
