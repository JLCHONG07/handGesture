from flask import Flask,Response
from flask import render_template
from hand import design
app = Flask(__name__)

#URL
app.add_url_rule('/','handDection',design.dectionPage)
app.add_url_rule('/page2','page2test',design.pageNo2)

app.add_url_rule('/page2/handCam','handCam',design.handRealtime)




if __name__=="__main__":
    app.run(debug=True)