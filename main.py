from flask import Flask,Response
from flask import render_template
from hand import classControl
app = Flask(__name__)


#URL
app.add_url_rule('/','home',classControl.homePage)
app.add_url_rule('/tryItOut','tryItOut',classControl.tryItOut)
app.add_url_rule('/tryItOut/handDetection','handDetection',classControl.handDetection)
app.add_url_rule('/tryItOut/hanDetection/handRealTimeCam','handRealTimeCam',classControl.handRealtime)



if __name__=="__main__":
    app.run(debug=True)