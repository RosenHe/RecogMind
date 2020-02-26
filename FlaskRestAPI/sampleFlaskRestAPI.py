# Using flask to make an api 
# import necessary libraries and functions 
from flask import Flask, jsonify, request, send_from_directory
from flask import render_template
import base64
from pathlib import Path
import numpy as np
import cv2
import time 
from keras.preprocessing import image

def getMyEmotion(imagePath):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    #-----------------------------
    #face expression recognizer initialization
    from keras.models import model_from_json
    model = None
    with open("facial_expression_model_structure.json", "r") as f:
        print(f)
        model = model_from_json(f.read())
    model.load_weights('facial_expression_model_weights.h5') #load weights

    a = 0
    b = 0
    c = 0

    #-----------------------------

    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')


    #ret, img = cap.read()
    img = cv2.imread(imagePath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    #print(faces) #locations of detected faces

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle to main image
        
        detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
        detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
        
        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        
        img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
        
        predictions = model.predict(img_pixels) #store probabilities of 7 expressions
        
        #find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
        max_index = np.argmax(predictions[0])
        
        emotion = emotions[max_index]

        if max_index == 3 or max_index == 2 or max_index ==5:
            a += 1
        elif max_index == 6 or max_index == 4:
            b += 1
        else:
            c += 1

        #write emotion text above rectangle
        cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        #process on detected face end
        #-------------------------

    #cv2.imshow('img',img)


    return_value = [a,b,c]
    maxE = return_value.index(max(return_value))
    print(maxE)

    cv2.destroyAllWindows()
    return maxE

def runSubProcessGetEmtoion(imagePath):
    from subprocess import run, PIPE
    p = run(['python', 'faceExpress.py', imagePath], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    rc = p.returncode
    lines = p.stdout.splitlines()
    for line in lines:
        strline = line.decode("utf-8")
        if "emotion value" in strline:
            return int(strline.split(":")[1].strip())

# creating a Flask app 
app = Flask(__name__, static_url_path='')

@app.route("/")
def index():
    return render_template("index.html")
  
# on the terminal type: curl http://127.0.0.1:5000/ 
# returns hello world when we use GET. 
# returns the data that we send when we use POST. 
@app.route('/getEmotionsData', methods = ['GET', 'POST']) 
def home():
    print(request)
    print(request.is_json)
    content = request.get_json(force=True)

    imageContent = content["image"]
    imageStr = imageContent.split(",")[1]
    image = base64.b64decode(imageStr)
    Path("image/").mkdir(parents=True, exist_ok=True)
    imagePath = "image/test.png"
    with open(imagePath, 'wb') as f:
        f.write(image)
    emotion = runSubProcessGetEmtoion(imagePath)
    print("My emotion is: " + str(emotion))
    emotionData = {
            "emotion": emotion,
            "0" : {"Music" :"https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlC?si=cSdiTgAyR2eu8JWC4x63Yg",
                   "Quotes" :"https://www.pinterest.com/rachelmaser/being-happy-quotes/",         
	               "Video" : "https://youtu.be/8KkKuTCFvzI",
                   "Articles" :"https://jamesclear.com/eat-healthy"},
            "1" : {"Music" : "https://open.spotify.com/playlist/37i9dQZF1DX7gIoKXt0gmx",
                  "Quotes" :"https://www.pinterest.com/search/pins/?rs=ac&len=2&q=uplifting%20quotes%20for%20hard%20times&eq=UPLIFTI&etslf=8059",
                  "Video" : "https://www.youtube.com/watch?time_continue=4&v=mYUQ_nlZgWE&feature=emb_title",
                  "Articles" :"https://www.lifehack.org/articles/communication/overcome-sadness-19-simple-things-you-didnt-realize-you-can.html"},
            "2" : {"Music" : "https://open.spotify.com/playlist/37i9dQZF1DX4WYpdgoIcn6",
                  "Quotes" :"https://www.pinterest.com/search/pins/?rs=ac&len=2&q=calming%20quotes&eq=CALMIN&etslf=7578",
                  "Video" : "https://www.youtube.com/watch?v=BsVq5R_F6RA",
                  "Articles" :"https://www.lifehack.org/articles/communication/20-things-when-you-feel-extremely-angry.html"}

            }
    return jsonify({'data': emotionData}) 

@app.route('/staticFiles/<path:path>')
def send_static(path):
    return send_from_directory('staticFiles', path)
  
# A simple function to calculate the square of a number 
# the number to be squared is sent in the URL when we use GET 
# on the terminal type: curl http://127.0.0.1:5000 / home / 10 
# this returns 100 (square of 10) 
@app.route('/test', methods = ['GET']) 
def disp(): 
    data = request.args.get("image")
    print(request.is_json)
    print(data)
    return jsonify(data) 
  
  
# driver function 
if __name__ == '__main__':
    app.run(debug=True) 