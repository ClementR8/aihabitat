
from ultralytics import YOLO
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import cv2


def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    """This function takes as parameters an image, the coordinates of a box containing the detected object on the image, the label of the detected object and color parameter.
    It returns the image with a box drawn around the detected object."""
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    # Convert image to a writable format
    image = cv2.UMat(image)
    
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    #cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    cv2.rectangle(image, (10,10), (0,0), (6, 112, 83), thickness=1, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0, lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)

    # Convert the image back to the NumPy array format
    image = cv2.UMat.get(image)
    return image


def draw_circle(image, center):
    """This function takes for parameters an image and the coordinates of a point at the center of an objects box.
    It draws a circle around a center point in the image."""
    image = cv2.UMat(image)
    cv2.rectangle(image, (10,10), (0,0), (6, 112, 83), thickness=1, lineType=cv2.LINE_AA)
    radius = 20
    color = (255, 255, 255)
    thickness = 2 
    cv2.circle(image, center, radius, color, thickness)
    image = cv2.UMat.get(image)
    return image


def all_bboxes(image, boxes, goal_object_label, labels=[], colors=[], score=True, conf=None):
  """This function uses the box_label function to draw the boxes on all the detected objects"""
  #Define labels
  if labels == []:
    labels = {0: u'__background__', 1: u'bed', 2: u'chair', 3: u'door', 4: u'door_frame', 5: u'shower', 6: u'sink', 7: u'sofa', 8: u'stairs', 9: u'table', 10: u'toilet'}
  #Define colors
  if colors == []:
    colors = [(89, 161, 197),(67, 161, 255),(19, 222, 24),(186, 55, 2),(167, 146, 11),(190, 76, 98),(130, 172, 179),(115, 209, 128),(204, 79, 135),(136, 126, 185),(209, 213, 45)]
  
  #plot each boxes
  cmpt = 0
  index = None 
  for box in boxes:
    #print(box)
    #print("box-1", int(box[-1]))
    if labels[int(box[-1])+1]==goal_object_label :
    	index = cmpt

    #add score in label if score=True
    if score :
      label = labels[int(box[-1])+1] + " " + str(round(100 * float(box[-2]),1)) + "%"
    else :
      label = labels[int(box[-1])+1]
    #filter every box under conf threshold if conf threshold setted
    if conf :
      if box[-2] > conf:
        color = colors[int(box[-1])]
        image = box_label(image, box, label, color) #modif
    else:
      color = colors[int(box[-1])]
      image =box_label(image, box, label, color) #modif
    cmpt += 1
    
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  
  return image, index, boxes


