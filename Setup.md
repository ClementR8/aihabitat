# Habitat Installation

To install the simulator habitat-sim, a tutorial called can be followed on the following link : [facebookresearch/habitat-sim:](https://github.com/facebookresearch/habitat-sim).
However, a few errors can occur while following the tutorial, so I’ll explain in detail the installation process.

1) Creating a new conda environment 
To install habitat you need conda on your computer. To do so you can follow the conda tutorial ([Installing conda — conda 24.4.1.dev80 documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)).
Assuming that conda is installed on your computer, we can create a new environment that we will call habitat. To do this type the following code lines in the terminal.
   ```bash
   # We require python>=3.9 and cmake>=3.10
   conda create -n habitat python=3.9 cmake=3.14.0
   conda activate habitat
   ```

2) Install habitat-sim in the environment
Once the new environment created, you can install habitat-sim in your environment. To do so several options are possible (see tutorial).
However, I chose the most common scenario, which is to install habitat-sim with bullet physics.
    ```
    conda install habitat-sim withbullet -c conda-forge -c aihabitat
    ```

By entering this command line, you will install the current version of habitat-sim. However, due to frequent changes of version in the simulator, some codes might not be working on the current version. To fix this you can work on a previous version and so install a previous version of habitat-sim on your computer. 
`conda install habitat-sim=0.1.6 -c conda-forge -c aihabitat` (to install version 0.3.0 of habitat-sim)
In my project I worked on the 0.3.0 version of habitat-sim.
Once you installed habitat-sim, you can check the version by typing the command line: 
pip show habitat-sim.

Right now habitat-sim is installed and you can start to launch end test new codes. However, it can be a good idea to learn with examples of codes ready to be tested. Follow the next part if you want to test examples but notice that following this part is not mandatory to try your codes. However, if you are just starting working on the simulator, I strongly advise you to test examples to understand them. 

3) Testing
A few options are possible to test example codes. 
First of all, I recommend you to follow the Testing part in the habitat-sim tutorial. 
For this part you will have to create a new folder in which you will download scenes and objects.
To download the testing scenes, use the following command : 
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path /path/to/data/
To download the example objects, use the following command : 
   ```bash
   python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path /path/to/data/
   ```
Then you might want to try some codes that are ready to be launched. Such examples are available on the habitat-sim github. You can download the viewer.py file as explained in the tutorial on the following link : habitat-sim/examples/viewer.py at main · facebookresearch/habitat-sim (github.com)
Put this file in a folder (called example in the following command), and write the following command in the terminal:
   ```bash
   python -m habitat_sim.utils.datasets_download --uids habitat_example_objects --data-path /path/to/data/
   ```
Note : Replace /path/to/data by the path to the folder in which the your habitat-test-scenes was downloaded (see above).

Pressing Enter should launch the code and a window should open. To move the agent inside the room you can use W/A/S/D keys to move forward/left/backward/right and arrow keys or mouse (LEFT click) to control gaze direction (look up/down/left/right).

Now you know how to test a code !

A lot of other examples are available on the habitat-sim Github ([facebookresearch/habitat-sim: A flexible, high-performance 3D simulator for Embodied AI research. (github.com)](https://github.com/facebookresearch/habitat-sim)) or on my aihabitat Github ([ClementR8/aihabitat: Repository containing setup codes and basic codes to have basic material and start using AiHabitat (github.com)](https://github.com/ClementR8/aihabitat)).
To try the codes you can download them one at a time or clone the repository.
For example you can type the following command line to copy my repository : 
   ```bash
   git clone https://github.com/ClementR8/aihabitat.git
   ```
Notice my repository uses scenes from the Habitat-Matterport 3D Research Dataset (HM3D), that are not on the repository because of their heavy storage. You need to download them in the following repository : [ matterport/habitat-matterport-3dresearch (github.com)](https://github.com/matterport/habitat-matterport-3dresearch). In my codes only the minival datasets are needed.


4) Good to know 
Each time you are working on a Linux terminal be careful of the environment you are working on. 
If you are working with habitat-sim, be sure you are working in the habitat environment.
Be also sure that you are working in the right directory. 
For example, if you are working on my cloned repository at the following address : /home/intern/habitat-sim/aihabitat ,
be sure that you are in the right directory, by using git command such as cd.
 
To launch your code you can type the command in two different ways :
   ```bash
   python basic_codes/room_changer.py 
   # OR
   python -m basic_codes.room_changer
   ```

Now that you know the basics, you can start to understand the codes and build your own !

