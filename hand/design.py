from flask import render_template, request, Response
import os
from hand.testing import start

def dectionPage():
    return render_template('videostream.html')

def pageNo2():
    return render_template('page2.html') 

def handRealtime():
   return Response(start(), mimetype='multipart/x-mixed-replace; boundary=frame')