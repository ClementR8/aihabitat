# ---
# jupyter:
#   accelerator: GPU
#   jupytext:
#     cell_metadata_filter: -all
#     formats: nb_python//py:percent,notebooks//ipynb
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.9.17
# ---

# %% [markdown]
# # Habitat-sim Interactivity
#
# This use-case driven tutorial covers Habitat-sim interactivity, including:
# - Adding new objects to a scene
# - Kinematic object manipulation
# - Physics simulation API
# - Sampling valid object locations
# - Generating a NavMesh including STATIC objects
# - Agent embodiment and continuous control

# %%
# @title Path Setup and Imports { display-mode: "form" }
# @markdown (double click to show code).

## [setup]
import math
import os
import random

import git
import magnum as mn
import numpy as np
import time

# %matplotlib inline
from matplotlib import pyplot as plt
from PIL import Image

import habitat_sim
from habitat_sim.utils import common as ut
from habitat_sim.utils import viz_utils as vut





def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.scene_dataset_config_file = settings["scene_dataset"]
    sim_cfg.enable_physics = settings["enable_physics"]

    # Note: all sensors must have the same resolution
    sensor_specs = []

    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [settings["height"], settings["width"]]
    color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(color_sensor_spec)

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(depth_sensor_spec)

    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(semantic_sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    if settings["action_space"] == 0:
	    agent_cfg.action_space = {
		"move_forward": habitat_sim.agent.ActionSpec(
		    "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
		),
		"turn_left": habitat_sim.agent.ActionSpec(
		    "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
		),
		"turn_right": habitat_sim.agent.ActionSpec(
		    "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
		),
	    }
    elif settings["action_space"] == 1:
	    agent_cfg.action_space = {
		"move_forward": habitat_sim.agent.ActionSpec(
		    "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
		),
		"turn_left": habitat_sim.agent.ActionSpec(
		    "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
		),
		"turn_right": habitat_sim.agent.ActionSpec(
		    "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
		),
		"incremental_right_turn": habitat_sim.agent.ActionSpec(
		    "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
		),
		"incremental_left_turn": habitat_sim.agent.ActionSpec(
		    "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
		),
	    }
    else : #elif settings["action_space"] == 2:
	    agent_cfg.action_space = {
		"move_forward": habitat_sim.agent.ActionSpec(
		    "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
		),
		"turn_left": habitat_sim.agent.ActionSpec(
		    "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
		),
		"turn_right": habitat_sim.agent.ActionSpec(
		    "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
		),
		"incremental_right_turn": habitat_sim.agent.ActionSpec(
		    "turn_right", habitat_sim.agent.ActuationSpec(amount=5.0)
		),
		"incremental_left_turn": habitat_sim.agent.ActionSpec(
		    "turn_left", habitat_sim.agent.ActuationSpec(amount=5.0)
		),
		"door_forward":habitat_sim.agent.ActionSpec(
		    "move_forward", habitat_sim.agent.ActuationSpec(amount=100)
		),
	    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])






