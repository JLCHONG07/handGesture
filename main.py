from flask import Flask, Response
from flask import render_template
from hand import classControl
app = Flask(__name__)


#URL
app.add_url_rule('/','home',classControl.homePage)
app.add_url_rule('/tryItOut','tryItOut',classControl.tryItOut)


app.add_url_rule('/tryItOut/handDetection','handDetection',classControl.handDetection)
app.add_url_rule('/tryItOut/hanDetection/handRealTimeCam','handRealTimeCam',classControl.handRealtime)
app.add_url_rule('/tryItOut/hanDetection/handRealTimeCam/handDetection2','handCamOnly', classControl.handCamOnly)
app.add_url_rule('/tryItOut/hanDetection/handRealTimeCam/handDetection2/handRealTimeCam2','handRealTimeCam2', classControl.handRealtime2)

app.add_url_rule('/tryItOut/mouse_controller_with_hand', 'mouse_controller_with_hand', classControl.mouse_controller_with_hand)
app.add_url_rule('/tryItOut/mouse_controller_with_hand/open_window_ui', 'open_window_ui', classControl.open_window_ui)

app.add_url_rule('/tryItOut/handRecognizeMode1','handRecognizeMode1',classControl.handModeRecognize1)
app.add_url_rule('/tryItOut/handRecognizeMode1/recognizeMode1','recognizeMode1',classControl.recognizeMode1)
app.add_url_rule('/tryItOut/handRecognizeMode1','handRecognizeMode2',classControl.handModeRecognize2)
app.add_url_rule('/tryItOut/handRecognizeMode1/recognizeMode2','recognizeMode2',classControl.recognizeMode2)


if __name__=="__main__":
    app.run(debug=True)