# DRL_5G_Schduling
Repo for the code of the 5G scheudling that is based on DRL.

To be able to run, you should run ./bash run_script.
The parser's information is provided in parser model under Misc_Folder.

# Setup
To run this code you need the following:

Python3,
Numpy,
Stable Baseline3,
Networkx,
loguru,
pandas,
argparse,
gym

# Training the model
Use the run_script.sh script to train the model. -ttt flag 1 indicates training phase and zero indictates evaluation phase. 

# Evaluating a saved model
You should define the checkpoints of the saved model. Load the model into the agent, and run the agent. 
You should define the number of episodes you wish to compare the greedy algorithm with the DRL agent.
You should define which DRL agent you wish to use in the Main_RL.py script, under method stable_agent.

# Citation
If you find this code useful please cite us in your work:


