

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

from setup.simulator_configuration import make_cfg



## We get the path to the current repository, the data and the output directory 

repo = git.Repo(".", search_parent_directories=True)
dir_path = repo.working_tree_dir
data_path = os.path.join(dir_path, "data")
output_directory = 	"examples/MyCodes/github_repository/videos/"  # @param {type:"string"}
output_path = os.path.join(dir_path, output_directory)
os.makedirs(output_path, exist_ok=True)

# Test scene
test_scene = os.path.join(data_path, "scene_datasets/mp3d_example/Example/GLAQ4DNUx5U.glb")
#test_scene = os.path.join(data_path, "scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb") # Another test scene 

# Scene dataset configuration files
mp3d_scene_dataset = os.path.join(data_path, "scene_datasets/mp3d_example/mp3d.scene_dataset_config.json")



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
    "seed": 1,  # used in the random navigation
    "enable_physics": False,  # kinematics only
    "action_space" : 0, # type of actions
}

cfg = make_cfg(sim_settings)
sim = habitat_sim.Simulator(cfg)

  
## Load the navmesh
# The navmesh defines which areas of the environment (scene) are traversable by an agent. It is then necessary if we want our agent to stay in the scene and not go through the scene frontiers (walls, floors, ...) 
sim.pathfinder.load_nav_mesh(
    os.path.join(data_path, "scene_datasets/mp3d_example/Example/GLAQ4DNUx5U.basis.navmesh")
)



action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
print("Discrete action space: ", action_names)


# the randomness is needed when choosing the actions
random.seed(sim_settings["seed"])
sim.seed(sim_settings["seed"])


## Agent Configuration

# Set agent state
agent = sim.initialize_agent(sim_settings["default_agent"])
agent_state = habitat_sim.AgentState()
agent_state.position = np.array([-0.6, 0.0, 0.0])  # position of the agent in the scene
agent.set_state(agent_state)

# Get agent state
agent_state = agent.get_state()
print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)


## Simulation of a Random Walk

time_step = 0.25

start_time = sim.get_world_time()
observations = []
video_prefix = "random-walk"


# Simulate for 10 seconds	
while sim.get_world_time() - start_time < 10.0:
    action = random.choice(action_names)
    #print("action", action)
    observations.append(sim.get_sensor_observations()) 
    sim.step(action)
    #sim.step_physics(0.25)
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




