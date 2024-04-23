
from ultralytics import YOLO
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import cv2


from setup.simulator_configuration import make_cfg
from image_processing.yolo_image_processing import box_label, draw_circle, all_bboxes



## [setup]
import math
import os
import random

import git
import magnum as mn
import numpy as np

# %matplotlib inline
from matplotlib import pyplot as plt
from PIL import Image

import habitat_sim
from habitat_sim.utils import common as ut
from habitat_sim.utils import viz_utils as vut


## We get the path to the current repository, the data and the output directory 

repo = git.Repo(".", search_parent_directories=True)
dir_path = repo.working_tree_dir
data_path = os.path.join(dir_path, "aihabitat/scenes")
output_directory = "videos/"  # @param {type:"string"}
output_path = os.path.join(dir_path, output_directory)
os.makedirs(output_path, exist_ok=True)
print(dir_path)
# Test scene
test_scene = os.path.join(data_path, "test-scenes/van-gogh-room.glb")
#test_scene = os.path.join(data_path, "scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb") # Another test scene 
#test_scene = os.path.join(data_path, "scene_datasets/mp3d_example/Example/GLAQ4DNUx5U.glb") # Another test scene 

# Scene dataset configuration files
mp3d_scene_dataset = os.path.join(data_path, "mp3d.scene_dataset_config.json")



## Simulator Configuration


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.add_argument("--no-make-video", dest="make_video", action="store_false")
    parser.set_defaults(show_video=True, make_video=True)
    args, _ = parser.parse_known_args()
    show_video = args.display
    display = args.display
    make_video = args.make_video
else:
    show_video = False
    make_video = False
    display = False

rgb_sensor = True  # @param {type:"boolean"}
depth_sensor = True  # @param {type:"boolean"}
semantic_sensor = True  # @param {type:"boolean"}


sim_settings = {
    "width": 256,  # Spatial resolution of the observations
    "height": 256,
    "scene": test_scene,  # Scene path
    "scene_dataset": mp3d_scene_dataset,  # the scene dataset configuration files
    "default_agent": 0,
    "sensor_height": 1.5,  # Height of sensors in meters
    "color_sensor": rgb_sensor,  # RGB sensor
    "depth_sensor": depth_sensor,  # Depth sensor
    "semantic_sensor": semantic_sensor,  # Semantic sensor
    "seed": 3,  # used in the random navigation
    "enable_physics": False,  # kinematics only
    "action_space" : 1, # type of actions
}

cfg = make_cfg(sim_settings)
sim = habitat_sim.Simulator(cfg)


## Definition of a function that will make the agent follow the object

def follow_object(image, x1, x2, epsilon = 10):
	size_x, size_y = np.shape(image)[0], np.shape(image)[1]
	#theta = np.arctan((size_x-x1-x2)/size_y)
	#print((x1+x2)/2)
	if ((x1+x2)/2 < size_x/2 - epsilon) : #Object on the left side of the screen
		action = "incremental_left_turn"
	elif ((x1+x2)/2 > size_x/2 + epsilon): 
		action = "incremental_right_turn"
	else :
		action = "move_forward"
	return action
  

## Action 


# List of actions
action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
print("Discrete action space: ", action_names)


# the randomness is needed when choosing the actions
random.seed(sim_settings["seed"])
sim.seed(sim_settings["seed"])



## Agent Configuration

# Set agent state
agent = sim.initialize_agent(sim_settings["default_agent"])
agent_state = habitat_sim.AgentState()
agent_state.position = np.array([-0.6, 0.0, 0.0])  # world space
agent.set_state(agent_state)

# Get agent state
agent_state = agent.get_state()
print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)


## Simulation of an object follower agent

time_step = 0.25

start_time = sim.get_world_time()
observations = []
video_prefix = "action"


#Define YOLO model
#model = YOLO("yolov8n.pt") # Pretrained YOLO model not accurate 
model_path = os.path.join(dir_path, "aihabitat/image_processing/yolo_model/weights/best.pt")
model = YOLO(model_path) # Custom YOLO model

#Object to find
object_to_find = "door"

	
# Simulate for 10 seconds
while sim.get_world_time() - start_time < 10.0:
    action = random.choice(action_names[0:2])
    #print("action", action)
    observation = sim.get_sensor_observations()
    #print(observation)
    #print(observation['color_sensor']) #There are 4 channels : R,G,B and alpha representing the transparency.
    image = observation['color_sensor'][:, :, :3]#Let's remove the transparency.
    results = model.predict(image) 
    #print(results)
    #print(results[0].boxes.data)
    boxed_image, index, boxes = all_bboxes(image, results[0].boxes.data, object_to_find, conf=0)
    #print("boxes : ", boxes)
    print("box: ", results[0].boxes.data)
    #print(boxed_image)
    if index is not None : # Goal object detected
    	print("Goal object detected")
    	print("Index : ", index)
    	print(results[0].boxes.data[index])
    	object_frame = results[0].boxes.data[index]
    	#print(results[0].boxes.data)
    	#print(results[0].boxes.data[0])
    	#print(object_frame)
    	x1, y1, x2, y2 = object_frame[0].item(), object_frame[1].item(), object_frame[2].item(), object_frame[3].item()
    	#print(image)
    	#image = cv2.UMat(image)
    	center = (int((x1+x2)/2), int((y1+y2)/2))
    	#print("Center : ", center)
    	boxed_image = draw_circle(boxed_image,center)
    	#cv2.circle(image, center, 1, "red")
    	#print(image)
    	#cv2.rectangle(image, center, (0,0), "red", thickness=1, lineType=cv2.LINE_AA)
    	action = follow_object(image, x1, x2)
    observation['color_sensor'][:, :, :3] = boxed_image
    observations.append(observation) 
    sim.step(action)
    sim.step_physics(0.25)
    #sim.step("move_forward")
    #print(sim.get_sensor_observations())
    

## Make video of simulation
  
vut.make_video(
            observations=observations,
            primary_obs="color_sensor",
            primary_obs_type="color",
            video_file=output_path + video_prefix,
            open_vid=show_video,
        )    





































"""
## Train + Save YOLO model on custom dataset COCO128

#model = YOLO("yolov8n.yaml")
#results = model.train(data="coco128.yaml", epochs=3)
#results = model.val(data="coco128.yaml")
#success = model.export(format="onnx")

## Predictions based on the model weights

model = YOLO("yolov8n.pt")

print("Blabla1")
#results = model.predict("https://images.unsplash.com/photo-1600880292203-757bb62b4baf?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2070&q=80")
image = cv2.imread("chair4.png")
results = model.predict("chair4.png")

print("Blabla2")
print(results)
plot_bboxes(image, results[0].boxes.data, conf=0)"""
