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
        static_image_mode=False,
        min_detection_confidence=0.75,
        max_num_hands=2,
        min_tracking_confidence=0.75)as hands:
    
     while cap.isOpened():
        success, image = cap.read()
        if not success:
         print("Ignoring empty camera frame.")
         cv2.destroyAllWindows()
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
        static_image_mode=False,
        min_detection_confidence=0.75,
        max_num_hands=2,
        min_tracking_confidence=0.75)as hands:
    
     while cap.isOpened():
        success, image = cap.read()
        if not success:
         print("Ignoring empty camera frame.")
         cv2.destroyAllWindows()
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
        #font = cv2.FONT_HERSHEY_PLAIN
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        #print(f'FPS:{int(fps)}')

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        #make it show only joint of hands
        image=np.zeros(image.shape[:],dtype="uint8")
        image= cv2.putText(image, f'FPS:{int(fps)}', (10, 30), cv2.FONT_HERSHEY_PLAIN, 3, (100, 255, 0), 3)
        if results.multi_hand_landmarks:
            for hand_landmarks,handedness in zip(results.multi_hand_landmarks,results.multi_handedness):
                #Loop for landmarks of the hand which able to adding the landmarks on the default lamdmarks
                #for id, lm in enumerate(hand_landmarks.landmark):
                    #print(id,lm)
                    #h,w,c=image.shape
                    #cx,cy=int(lm.x*w),int(lm.y*h)
                    #print(id,cx,cy)
                    #cv2.circle(image,(cx,cy),10,(255,0,255),cv2.FILLED)
                                # Calculate the circumscribed rectangle
                #left hand coordinate
                cx,cy=hand_coordinate(image,hand_landmarks,handedness)
                #right hand coordinate
                cx2,cy2=hand_coordinate2(image,hand_landmarks,handedness)
                image=cv2.putText(image, f'Left : {str(cx)}  {str(cy)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (100, 255, 0), 3)
                image=cv2.putText(image, f'Right : {str(cx2)}  {str(cy2)}', (10, 110), cv2.FONT_HERSHEY_PLAIN, 3, (100, 255, 0), 3)
                #image=cv2.putText(image, f'{str(cy)}', (50, 80), font, 3, (100, 255, 0), 3)
                #calculation of box surrounded hand
                brect = calc_bounding_rect(image, hand_landmarks)
                #draw box surrounded hand
                image = draw_bounding_rect(True, image, brect)
                #words of left and right
                image = draw_info_text(
                    image,
                    brect,
                    handedness
                )
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

     
def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
         # External Rectangle
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (255,0,255), 2)

    return image

def draw_info_text(image, brect, handedness):
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return image

def hand_coordinate(image,hand_landmarks,handedness):
    cxID0=0
    cyID0=0
    cxID9=0
    cyID9=0
    posxLeft=0
    posyLeft=0
    info_text = handedness.classification[0].label[0:]
    #print(info_text)
    if info_text=="Left":
        
        for id, lm in enumerate(hand_landmarks.landmark):
            #print("Left Hand :")
            #print(id,lm)
            h,w,c=image.shape
            cx,cy=int(lm.x*w),int(lm.y*h)
            if id==0:
                cxID0,cyID0=cx,cy
            if id==9:
                cxID9,cyID9=cx,cy
    
        posxLeft=(cxID0+cxID0)/2
        posyLeft=(cyID0+cyID9)/2

    
        return posxLeft,posyLeft

    else:
        return "",""

def hand_coordinate2(image,hand_landmarks,handedness):
    cxID0=0
    cyID0=0
    cxID9=0
    cyID9=0
    posxRight=0
    posyRight=0
    info_text = handedness.classification[0].label[0:]
    if info_text=="Right":
        
        for id, lm in enumerate(hand_landmarks.landmark):
            #print("Right Hand :")
            #print(id,lm)
            h,w,c=image.shape
            cx,cy=int(lm.x*w),int(lm.y*h)
            if id==0:
                cxID0,cyID0=cx,cy
            if id==9:
                cxID9,cyID9=cx,cy
    
        posxRight=(cxID0+cxID0)/2
        posyRight=(cyID0+cyID9)/2

    
        return posxRight,posyRight

    else:
        return "",""

                
            

