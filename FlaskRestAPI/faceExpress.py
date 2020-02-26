import numpy as np
import cv2
import time 
from keras.preprocessing import image
import sys

#-----------------------------
#opencv initialization

def getMyEmotion(imagePath):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    #-----------------------------
    #face expression recognizer initialization
    from keras.models import model_from_json
    model = model_from_json(open("facial_expression_model_structure.json", "r").read())
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
    print('The emotion value is:',maxE)
    cv2.destroyAllWindows()

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))
imagePath = ""
if(len(sys.argv) > 0):
    imagePath = sys.argv[1]
else:
    imagePath = "image/test.png"
getMyEmotion(imagePath)