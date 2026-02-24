from roboflow import Roboflow
import inference
import cv2
import supervision as sv
from ultralytics import YOLO
from ultralytics import YOLOWorld
import os
from inference.models.utils import get_model
from PIL import Image


# YOLO world model example

model = YOLOWorld('yolov8x-worldv2.pt')
model.set_classes(['milk','butter','plate','knife','spoon','orange juice','glass','table','chair','croissant','bun','bread','cheese','meat','egg','mug','coffee','cup'])

directory = './breakfast/'
for f in os.listdir(directory):
    if (f.endswith('.jpg') or f.endswith('.jpeg')):
       results = model.predict(directory+f) #, conf=0.5)
       results[0].save('./predictions/'+f)

# Roboflow trained model

# add roboflow API key here
rfapikey = ''

rf = Roboflow(api_key=rfapikey)

# change workspace and project name to what you created 
project = rf.workspace('ardis').project('ardis')
model = project.version(1).model

directory = './test-collection/'

i = 'wekit2.jpg'
results = model.predict('./test-collection/'+i, confidence=15, overlap=30).json()
print(results)
model.predict(directory+i, confidence=0.15, overlap=0.30).save("predictions/"+i)

i = 'wekit1.jpg'
results = model.predict('./test-collection/'+i, confidence=15, overlap=30).json()
print(results)
model.predict('./test-collection/'+i, confidence=15, overlap=30).save("predictions/"+i)

