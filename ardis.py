from roboflow import Roboflow
import inference
import cv2
import supervision as sv
from ultralytics import YOLO
from ultralytics import YOLOWorld
import os
from inference.models.utils import get_model
from PIL import Image

# EXAMPLE with YOLOv8x world model

#model = YOLO('yolov8l-world.pt')
#model = YOLOWorld('yolov8s-world.pt')
model = YOLOWorld('yolov8x-worldv2.pt')

#model.set_classes(["person", "dog", "eye", "tongue"])
#model.set_classes(['bed','flacon','sink','table','screen','power socket'])
model.set_classes(['milk','butter','plate','knife','spoon','orange juice','glass','table','chair','croissant','bun','bread','cheese','meat','egg','mug','coffee','cup'])

directory = './test-collection/breakfast/'
for f in os.listdir(directory):
    if (f.endswith('.jpg') or f.endswith('.jpeg')):
       #image = cv2.imread(f)
       results = model.predict(directory+f) #, conf=0.5)
       results[0].save('./predictions/'+f)
       #results[0].show()

# EXAMPLE with own trained model from 73 pictures of ESAOTE machine
# (with mAP=96.7% for test set) 

rf = Roboflow(api_key='9c5m7EGgqJcCl7dgLvZS')
project = rf.workspace('ardis').project('ardis')
model = project.version(1).model

#model = get_model(model_id='ardis/1', api_key='9c5m7EGgqJcCl7dgLvZS')
#model = inference.get_roboflow_model("ardis/1", api_key='9c5m7EGgqJcCl7dgLvZS')

directory = './test-collection/'

i = 'wekit2.jpg'
results = model.predict('./test-collection/'+i, confidence=15, overlap=30).json()
print(results)
model.predict(directory+i, confidence=0.15, overlap=0.30).save("predictions/"+i)
#img = Image.open('predictions/'+i)
#img.show()

i = 'wekit1.jpg'
results = model.predict('./test-collection/'+i, confidence=15, overlap=30).json()
print(results)
model.predict('./test-collection/'+i, confidence=15, overlap=30).save("predictions/"+i)
#img = Image.open('predictions/'+i)
#img.show()


# OLD code four boundary box marking with infer

#detections = sv.Detections.from_inference(results[0])
#bounding_box_annotator = sv.BoundingBoxAnnotator()
#label_annotator = sv.LabelAnnotator()
#labels = [classes[class_id] for class_id in detections.class_id]
#annotated_image = bounding_box_annotator.annotate(
#    scene=image, detections=detections
#)
#annotated_image = label_annotator.annotate(
#    scene=annotated_image, detections=detections, labels=labels
#)
#sv.plot_image(annotated_image)

