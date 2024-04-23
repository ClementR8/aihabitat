
from ultralytics import YOLO
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import cv2


## Define base model 

model = YOLO("yolov8n.pt")


## Train model 

#Train on custom dataset (dataset build and annotated for the purpose of this project)
results = model.train(data="/home/intern/habitat-sim/examples/MyCodes/github_repository/image_processing/data/data.yaml", epochs=300)
results = model.val(data="/home/intern/habitat-sim/examples/MyCodes/github_repository/image_processing/data/data.yaml")
success = model.export(format="onnx") # Save model


##Train on COCO dataset (famous dataset for yolo training)
#results = model.train(data="coco128.yaml", epochs=3)
#results = model.val(data="coco128.yaml")
#success = model.export(format="onnx") # Save model
