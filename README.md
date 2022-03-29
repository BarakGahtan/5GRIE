# 5G Scheduling
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
@misc{https://doi.org/10.48550/arxiv.2203.14790,
  doi = {10.48550/ARXIV.2203.14790}, 
  url = {https://arxiv.org/abs/2203.14790},
  author = {Gahtan, Barak},
  keywords = {Networking and Internet Architecture (cs.NI), Machine Learning (cs.LG), Systems and Control (eess.SY), FOS: Computer and information sciences, FOS: Computer and information sciences, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering},
  title = {5G Routing Interfered Environment},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}


