import cv2
import mediapipe as mp
import numpy as np
import time
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands



def openCam():
    prev_frame_time=0
    
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
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)
        #print(results)
        
        #make fps font + formula 
        font = cv2.FONT_HERSHEY_SIMPLEX
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        #print(f'FPS:{int(fps)}')

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image= cv2.putText(image, f'FPS:{int(fps)}', (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
              mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        #cv2.imshow('MediaPipe Hands', image)
        #return image
        #self.frameOnWeb(image)    
        #if cv2.waitKey(5) & 0xFF == 27:
            #break

        #build frame/cam on web
        frame = cv2.imencode('.jpg', image)[1].tobytes()
        yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'

def secondMode():
    prev_frame_time=0
    
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
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)
        #print(results)

        #make fps font + formula 
        font = cv2.FONT_HERSHEY_SIMPLEX
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        #print(f'FPS:{int(fps)}')

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image= cv2.putText(image, f'FPS:{int(fps)}', (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

        #make it show only joint of hands
        mask=np.zeros(image.shape[:],dtype="uint8")
        image= cv2.putText(mask, f'FPS:{int(fps)}', (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
              mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        #cv2.imshow('MediaPipe Hands', image)
        #return image
        #self.frameOnWeb(image)    
        #if cv2.waitKey(5) & 0xFF == 27:
            #break

        #build frame/cam on web
        frame = cv2.imencode('.jpg', image)[1].tobytes()
        yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'

     
