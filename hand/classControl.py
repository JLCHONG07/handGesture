from flask import render_template, request, Response
import os
from hand.handmesh import openCam

def homePage():
    return render_template('home.html')

def tryItOut():
    return render_template('tryItOut.html')

def handDetection():
    return render_template('handDetection.html') 

def handRealtime():
   return Response(openCam(), mimetype='multipart/x-mixed-replace; boundary=frame')