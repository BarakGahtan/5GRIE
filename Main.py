import copy
import math
import os

import networkx as nx
import numpy as np
import torch
from loguru import logger

from Baseline_Benchmark.greedy_baseline import greedy_baseline
from Environment.Env import NetworkEnv
from Environment.Interfernce_Model.Interference_Maker import Interference
from Environment.Packets import Packets
from Environment.Network_Topology import Network
from RL_dir.Main_RL import RunAgents
from Misc_Folder import Constants as c
from Misc_Folder.Parser import Parser
from Misc_Folder.Utils_Session import get_session

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf


def make_name(x, y):
    return str(x) + "_" + str(y)

#Generate the list for comparison, so the greedy algorithm and DRL algorithms will use the same data.
@logger.catch
def generate_state_for_comparison(net, HP, opts, shortest_path_list, id_to_edges, edges_to_id, episodes=None):
    demand_episodes = []
    if episodes is not None:
        counter = episodes
    else:
        counter = int(HP['number_of_episodes'])
    for count in range(counter):
        next_episode_demand_matrix = copy.deepcopy(np.zeros(shape=(len(list(net.graph_topology.nodes)), len(list(
            net.graph_topology.nodes)))))  # empty matrix for the next observation.
        if opts.run_scene == "train":
            next_episode_demand_matrix = np.full(next_episode_demand_matrix.shape, opts.flows)
        else:
            next_episode_demand_matrix[0] = np.full(next_episode_demand_matrix.shape[0], opts.flows)
            next_episode_demand_matrix[:, 0] = opts.flows
        np.fill_diagonal(next_episode_demand_matrix, 0)
        for i in range(next_episode_demand_matrix.shape[0]):
            for j in range(next_episode_demand_matrix.shape[1]):
                if j not in shortest_path_list[i]:
                    next_episode_demand_matrix[i][j] = 0
        total_counter, Global_counter = 0, 0
        flows = []
        for i in range(next_episode_demand_matrix.shape[0]):
            for j in range(next_episode_demand_matrix.shape[1]):
                sum_packets_from_i_to_j = 0
                for cnt in range(int(next_episode_demand_matrix[i][j])):
                    # num_of_packets = np.random.uniform(low=opts.low_count_packets, high=opts.high_count_packets)
                    num_of_packets = opts.packets_amount
                    if len(shortest_path_list[i]) == 1:
                        next_episode_demand_matrix[i][j] = 0
                        continue
                    if j in shortest_path_list[i]:
                        flows.insert(0,
                                     copy.deepcopy(
                                         Packets(shortest_path_list, given_flow_id=Global_counter, source=i,
                                                 destination=j, size=num_of_packets, name=str(i) + "_" + str(j))))
                    else:
                        next_episode_demand_matrix[i][j] = 0
                    Global_counter += 1
                    sum_packets_from_i_to_j += num_of_packets
                next_episode_demand_matrix[i][j] = sum_packets_from_i_to_j
        total_counter = next_episode_demand_matrix.sum()
        interference_model = Interference(net, id_to_edges, edges_to_id, opts)
        if 0 <= count < counter/19:
            interference_model.make_interference_model(True, opts.eval_numbers[0])
        elif math.ceil(counter/19) <= count < 2*(counter/19):
            interference_model.make_interference_model(True, opts.eval_numbers[1])
        elif math.ceil(2*(counter/19)) <= count < 3*(counter/19):
            interference_model.make_interference_model(True, opts.eval_numbers[2])
        elif math.ceil(3*(counter/19)) <= count < 4*(counter/19):
            interference_model.make_interference_model(True, opts.eval_numbers[3])
        elif math.ceil(4*(counter/19)) <= count < 5*(counter/19):
            interference_model.make_interference_model(True, opts.eval_numbers[4])
        elif math.ceil(5*(counter/19)) <= count < 6*(counter/19):
            interference_model.make_interference_model(True, opts.eval_numbers[5])
        elif math.ceil(6*(counter/19)) <= count < 7*(counter/19):
            interference_model.make_interference_model(True, opts.eval_numbers[6])
        elif math.ceil(7*(counter/19)) <= count < 8*(counter/19):
            interference_model.make_interference_model(True, opts.eval_numbers[7])
        elif math.ceil(8*(counter/19)) <= count < 9*(counter/19):
            interference_model.make_interference_model(True, opts.eval_numbers[8])
        elif math.ceil(9*(counter/19)) <= count < 10*(counter/19):
            interference_model.make_interference_model(True, opts.eval_numbers[9])
        elif math.ceil(10*(counter/19)) <= count < 11*(counter/19):
            interference_model.make_interference_model(True, opts.eval_numbers[10])
        elif math.ceil(11*(counter/19)) <= count < 12*(counter/19):
            interference_model.make_interference_model(True, opts.eval_numbers[11])
        elif math.ceil(12*(counter/19)) <= count < 13*(counter/19):
            interference_model.make_interference_model(True, opts.eval_numbers[12])
        elif math.ceil(13*(counter/19)) <= count < 14*(counter/19):
            interference_model.make_interference_model(True, opts.eval_numbers[13])
        elif math.ceil(14*(counter/19)) <= count < 15*(counter/19):
            interference_model.make_interference_model(True, opts.eval_numbers[14])
        elif math.ceil(15*(counter/19)) <= count < 16*(counter/19):
            interference_model.make_interference_model(True, opts.eval_numbers[15])
        elif math.ceil(16*(counter/19)) <= count < 17*(counter/19):
            interference_model.make_interference_model(True, opts.eval_numbers[16])
        elif math.ceil(17*(counter/19)) <= count < 18*(counter/19):
            interference_model.make_interference_model(True, opts.eval_numbers[17])
        else:
            interference_model.make_interference_model(True, opts.eval_numbers[18])

        demand_episodes.append((copy.deepcopy(next_episode_demand_matrix), copy.deepcopy(total_counter),
                                copy.deepcopy(flows), copy.deepcopy(interference_model)))
    return demand_episodes

def create_list_from_file(file_name):
    # Using readlines()
    file1 = open(file_name, 'r')
    Lines = file1.readlines()
    greedy_steps, greedy_left_data, dropped_packets = [], [], []
    count = 0
    # Strips the newline character
    for line in Lines:
        if line.startswith("steps list"):
            greedy_steps = [int(i) for i in list(line.split(":")[1][1:-2].split(","))]
        if line.startswith("left data"):
            greedy_left_data = [int(i) for i in list(line.split(":")[1][1:-2].split(","))]
        if line.startswith("["):
            dropped_packets.append(float(line.split(":")[3]))
        count += 1
    return greedy_steps, greedy_left_data, dropped_packets


def main():
    parsed_args = Parser()
    opts = parsed_args.parse()
    sess = get_session()
    summary_writer = tf.summary.create_file_writer("./tensorboard_x_")
    HP = c.init_constants(opts)
    simulated_network = Network(opts, HP['rings'])
    observation_mat, edges_info = simulated_network.network_create()
    shortest_path_list = dict(nx.all_pairs_shortest_path(simulated_network.graph_topology))  # shortest_path_list[0][6] - shortest path from 0->6
    result = [make_name(x, y) for x, y in zip(edges_info['source'], edges_info['target'])]
    edges_to_id = {result[x]: x for x in range(len(result))}  #edges to id
    id_to_edges = {x: result[x] for x in range(len(result))} #id to edges
    interference_model = Interference(simulated_network, id_to_edges, edges_to_id, opts)
    interference_model.make_interference_model(False, 0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    demand_episodes_list_train = generate_state_for_comparison(simulated_network, HP, opts, shortest_path_list,
                                                               id_to_edges, edges_to_id, opts.episode_compare)
    demand_episode_evaluate = generate_state_for_comparison(simulated_network, HP, opts, shortest_path_list,
                                                            id_to_edges, edges_to_id, opts.episode_eval)
    network_environment = NetworkEnv(net=simulated_network, edges_info=edges_info, shortest_path_mat=shortest_path_list,
                                     edges_dict=edges_to_id, opts=opts, demand_list=demand_episodes_list_train,
                                     interference=interference_model, dev=device, eval_episodes=demand_episode_evaluate,
                                     id_to_edges=id_to_edges, edges_to_id=edges_to_id)
    steps_greedy, greedy_left_data, dropped_packets = [], [], []
    if opts.baseline == 1:
        greedy_baseline_ = greedy_baseline(simulated_network, edges_info, shortest_path_list, interference_model, opts,
                                           network_environment, id_to_edges, edges_to_id)
        data_greedy, steps_greedy, done_greedy, greedy_left_data = greedy_baseline_.greedy_baseline_method(opts)
    else:  # No need to generate a baseline run.
        steps_greedy, greedy_left_data,dropped_packets = create_list_from_file(opts.greedy_file_name)
    if len(steps_greedy) > 0:
        avg_steps = sum(steps_greedy) / len(steps_greedy)
    else:
        avg_steps = 0
    env_trainers = RunAgents(network_environment, HP, opts, avg_steps, steps_greedy, opts.train_single, greedy_left_data,dropped_packets)

if __name__ == "__main__":
    main()
