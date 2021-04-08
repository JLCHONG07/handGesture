from flask import render_template, request, Response
import os
from hand.handmesh import openCam, secondMode
from mouse_controller_with_hand_gesture import mouse_controller as mc
from hand.handRecognition import rmStartMode1

def homePage():
    return render_template('home.html')

def tryItOut():
    return render_template('tryItOut.html')

def handDetection():
    return render_template('handDetection.html') 

def handRealtime():
   return Response(openCam(), mimetype='multipart/x-mixed-replace; boundary=frame')

def handCamOnly():
    return render_template('handOnlyMode.html') 


def handRealtime2():
    return Response(secondMode(), mimetype='multipart/x-mixed-replace; boundary=frame')


def mouse_controller_with_hand():
    return render_template('mouse_controller.html')


def open_window_ui():
    return Response(mc.main())


def handModeRecognize1():
    return render_template('handRecogize.html')

def recognizeMode1():
    return Response(rmStartMode1(),mimetype='multipart/x-mixed-replace; boundary=frame')