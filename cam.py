
import numpy as np
import requests

import os
import random
from  tensorflow.keras.preprocessing import  image
from tensorflow import keras
import  tensorflow as tf
from  tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import categorical_crossentropy
from  tensorflow.keras.layers import Dense,MaxPool2D,Conv2D,Activation,Flatten,BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
import glob
import matplotlib.pyplot as plt
import itertools

from flask import Flask ,Response, request,template_rendered,render_template

##########################################

mobile=tf.keras.applications.mobilenet.MobileNet()

####################################



###############
key = 'f5fuNcsPy2ew11gimnI7yhE8vY3ujzh9E61ogmtO'
orignial = "https://api.nal.usda.gov/fdc/v1/foods/search?query=mango&pageSize=2&api_key=f5fuNcsPy2ew11gimnI7yhE8vY3ujzh9E61ogmtO"


###############

classes=[]
classes1=[]

nutration=[]
name_nutration = []
uni = []
gram = []
discription=[]
final=[]


###########

with open('fruits_name.txt',encoding="utf8") as f:
    classes1 = [line.strip() for line in f.readlines()]


classes = [x.lower() for x in classes1]
################

def find_details(name):

    food_name = str(name)
    path = f'https://api.nal.usda.gov/fdc/v1/foods/search?query={food_name}&pageSize=2&api_key=f5fuNcsPy2ew11gimnI7yhE8vY3ujzh9E61ogmtO'
    response = requests.get(path)
    data = response.json()

    k = 'foodSearchCriteria'

    k = data['foods']
    w = k[0]



    # print(w['foodNutrients'])
    discription.append(w['description'])
    findall = w['foodNutrients']

    nutration = 'nutrientName'
    unitname = 'unitName'
    value = 'value'
    for i in findall:
        name_nutration.append(i[nutration])
        uni.append(i[unitname])
        gram.append(i[value])





###########

def mob(img):
    img = image.load_img('apple.jpg', target_size=(224, 224))
    img_arry = image.img_to_array(img)
    img_arry_expanded_dims = np.expand_dims(img_arry, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_arry_expanded_dims)



###############
# orignal_image=mob()
# #
# predect=(orignal_image)
# predict=mobile.predict(predect)

# result=tf.keras.applications.imagenet_utils.decode_predictions(predict)


# find_food=result[0][0][1]
# probability=result[0][0][2]


############

def filter_name(name):
    name=name.replace("_"," ")
    name=name.lower()
    return name












###########

app=Flask(__name__,template_folder='camera')


import cv2 as cv

import datetime,time
from threading import Thread

import os

global frame

camer=cv.VideoCapture(0)


try:
    os.makedirs('images_file')
except OSError as error:
    pass    




#####################################################################################






#####################################################################################





def gen():
    while True:
       sucess,frame=camer.read()
       ret, buffer = cv.imencode('.jpg', cv.flip(frame,1))
       frame=buffer.tobytes()
       yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

      
                









def capture():
    sucess,frame=camer.read()
    cv.imwrite('images_file/after.jpg',frame)
    
    


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
   if request.method=='POST':
       if request.form.get('click')=='Capture':
           capture()
           camer.release()
           cv.destroyAllWindows()
           img=cv.imread('images_file/after.jpg')
           original_image=mob(img)
           predect=(original_image)
           predict=mobile.predict(predect)
           result=tf.keras.applications.imagenet_utils.decode_predictions(predict)
           find_food1=result[0][0][1]
           probability=result[0][0][2]
           find_food=filter_name(find_food1)
           if find_food in classes:
               find_details(find_food)
               for i,j,k in zip(name_nutration,uni,gram):
                   final.append((i,j,k))
               despti=str(discription[0])    
               return render_template('main.html',final=final,name=despti)    



           else:
               return render_template('error.html')    




       if request.form.get('click')=='Import':
           camer.release()
           cv.destroyAllWindows()
           return render_template('after.html')  



          
@app.route('/after',methods=["GET","POST"])
def after():
    img=request.files['file']
    img.save('images_file/after.jpg')
    img.save('static/file12.jpg')
    img=cv.imread('images_file/after.jpg')
    original_image=mob(img)
    predect=(original_image)
    predict=mobile.predict(predect)
    result=tf.keras.applications.imagenet_utils.decode_predictions(predict)
    find_food1=result[0][0][1]
    probability=result[0][0][2]
    find_food=filter_name(find_food1)
    if find_food in classes:
            find_details(find_food)
            for i,j,k in zip(name_nutration,uni,gram):
                   final.append((i,j,k))
            discripti=str(discription[0])       
            return render_template('main.html',name=discripti,final=final)
    else:
        return render_template('error.html')




    



    

if __name__=="__main__":
    app.run(port=3000,debug=True)