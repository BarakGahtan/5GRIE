import collections
import copy
import math

import gym
import numpy as np
import pandas as pd
from loguru import logger

from Environment.Interfernce_Model.Interference_Maker import Interference
from Environment.Packets import Packets
from Environment.Stations import Stations


class NetworkEnv(gym.Env):
    def __init__(self, net, edges_info, shortest_path_mat, edges_dict, opts, demand_list, interference, dev,
                 eval_episodes, id_to_edges, edges_to_id):
        super().__init__()
        self.edges_to_id = copy.deepcopy(edges_to_id)
        self.id_to_edges = copy.deepcopy(id_to_edges)
        self.dict_param = opts.__dict__
        self.opts = opts
        self.step_count = 0
        self.net = net
        self.episode_count = 0
        self.dropped_packets = 0
        self.device = dev
        self.directed_edges = list(net.graph_topology.edges)
        self.global_all_shortest_path = shortest_path_mat
        self.edges_info = pd.DataFrame(edges_info)
        self.original_network_object = net
        self.interference = interference
        self.next_episode_demand_matrix, self.total_counter, self.global_counter = 0, 0, 0
        self.routers_list = {}
        nodes = list(net.graph_topology.nodes)
        for i in range(len(nodes)):
            self.routers_list[str(nodes[i])] = Stations(name=str(nodes[i]), network_topology=self.original_network_object, shortest_path_list=shortest_path_mat[nodes[i]],
                                                        edges_dict=edges_dict, transceiver=self.dict_param['transceiver_count'], queue_size=self.dict_param['queue_size'])
        self.episode_demand_train = demand_list
        self.episode_demand_eval = eval_episodes
        for r in self.routers_list:
            self.routers_list[r].get_buffers_observations()
        numpy_obs = self.get_state_observation()
        self.observation_space = gym.spaces.Box(low=0,
                                                shape=np.concatenate((numpy_obs.flatten(),
                                                                      self.interference.interference_matrix_edges.flatten())).shape,
                                                high=1,
                                                dtype=np.float32)  # observation space
        # self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.interference.interference_matrix_edges.shape[0],)) #according to the number of links in the system
        self.action_space = gym.spaces.MultiDiscrete([101 for _ in range(self.interference.interference_matrix_edges.shape[0])])  # number of power level (steps)
        # self.action_space = gym.spaces.MultiBinary(self.interference.interference_matrix_edges.shape[0])

        logger.info(f"Environment finished init")

    # generate a ramdom demand matrix of pairs of source and destination from a uniform distributions for the training phase.
    # compatible with Stable Baselines3 convention.
    def generate_demand_random(self):
        next_episode_demand_matrix = copy.deepcopy(np.zeros(shape=(len(list(self.net.graph_topology.nodes)), len(list(
            self.net.graph_topology.nodes)))))  # empty matrix for the next observation.
        next_episode_demand_matrix = np.random.randint(low=self.dict_param['low_count_flows'],
                                                       high=self.dict_param['high_count_flows'],
                                                       size=next_episode_demand_matrix.shape)
        np.fill_diagonal(next_episode_demand_matrix, 0)
        for i in range(next_episode_demand_matrix.shape[0]):
            for j in range(next_episode_demand_matrix.shape[1]):
                if j not in self.global_all_shortest_path[i]:
                    next_episode_demand_matrix[i][j] = 0
        total_counter, self.global_counter = 0, 0
        flows = []
        for i in range(next_episode_demand_matrix.shape[0]):
            for j in range(next_episode_demand_matrix.shape[1]):
                sum_packets_from_i_to_j = 0
                for cnt in range(int(next_episode_demand_matrix[i][j])):
                    num_of_packets = np.random.randint(low=self.dict_param['low_count_packets'],
                                                       high=self.dict_param['high_count_packets'])
                    if len(self.global_all_shortest_path[i]) == 1:
                        next_episode_demand_matrix[i][j] = 0
                        continue
                    if j in self.global_all_shortest_path[i]:
                        flows.insert(0, copy.deepcopy(
                            Packets(self.global_all_shortest_path, given_flow_id=self.global_counter,
                                    source=i,
                                    destination=j, size=num_of_packets,
                                    name=str(i) + "_" + str(j))
                        )
                                     )
                    else:
                        next_episode_demand_matrix[i][j] = 0
                    self.global_counter += 1
                    total_counter += num_of_packets
                    sum_packets_from_i_to_j += num_of_packets
                next_episode_demand_matrix[i][j] = sum_packets_from_i_to_j

        interference_model = Interference(self.net, self.id_to_edges, self.edges_to_id, self.opts)
        interference_model.make_interference_model(False, 0)
        self.interference = copy.deepcopy(interference_model)
        self.total_counter = total_counter
        self.first_total_counter = total_counter
        self.next_episode_demand_matrix = next_episode_demand_matrix
        for f in flows:
            amount = self.routers_list[str(f.source)].add_flow(copy.deepcopy(f))
            self.total_counter -= amount

    def generate_demand_comparison(self, next_episode_demand, idx):
        # creating a data demand matrix for the next frame ( episode )
        self.next_episode_demand_matrix = copy.deepcopy(next_episode_demand[idx][0])
        self.first_total_counter = copy.deepcopy(next_episode_demand[idx][1])
        self.total_counter = copy.deepcopy(next_episode_demand[idx][1])
        self.next_episode_demand_matrix = np.divide(self.next_episode_demand_matrix, self.total_counter)
        self.interference = copy.deepcopy(next_episode_demand[idx][3])  # interference
        flows = copy.deepcopy(next_episode_demand[idx][2])
        for f in flows:
            amount = self.routers_list[str(f.source)].add_flow(copy.deepcopy(f))
            self.total_counter -= amount

    def process_flows(self):
        flows = []
        for name, router in self.routers_list.items():
            flows += self.routers_list[name].get_all_flows()
        counter = 0
        for flow in flows:
            current_location, next_location = flow.current_location, flow.next_hop
            link = str(current_location) + "_" + str(next_location)
            if math.ceil(self.routers_list[str(current_location)].is_link_activated(link)) == 1:
                isFinished, amountOrTemp = self.routers_list[str(current_location)].remove_flow(flow, self.total_counter)
                if isFinished == 0:
                    counter += amountOrTemp.size
                    amount = self.routers_list[str(amountOrTemp.current_location)].add_flow(amountOrTemp)
                    self.total_counter -= amount
                elif isFinished == -1:  # nothing happened bw was full
                    continue
                elif isFinished == 1:  # a flow finished, either partial or whole
                    self.total_counter -= amountOrTemp
                    counter += amountOrTemp
                else:
                    print("SHOULD NOT GET HERE")
        flows_check = []
        for name, router in self.routers_list.items():
            flows_check += self.routers_list[name].get_all_flows()
        return len(flows_check) == 0, self.total_counter

    def get_dropped_packets(self):
        dropped_packets = 0
        for r in self.routers_list.keys():
            dropped_packets += self.routers_list[r].get_dropped_packets_in_router()
        return dropped_packets

    def get_state_observation(self):  # load of each router queues.
        numpy_list = []
        unified_dict = {}
        for router in self.routers_list:
            self.routers_list[router].get_buffers_observations()
            if len(self.routers_list[router].observation_moving_dropped_packets_prop) != 0:
                numpy_list.append(self.routers_list[router].observation_moving_dropped_packets_prop)
        unified_obs_dict = {k: v for d in numpy_list for k, v in d.items()}
        obs_numpy_array_packets = np.zeros(shape=(len(unified_obs_dict),))
        obs_numpy_array_load_link = np.zeros(shape=(len(unified_obs_dict),))
        obs_numpy_dropped_packets = np.zeros(shape=(len(unified_obs_dict),))
        for e in unified_obs_dict.keys():
            obs_numpy_array_packets[self.edges_to_id[e]] = unified_obs_dict[e][0]
            obs_numpy_array_load_link[self.edges_to_id[e]] = unified_obs_dict[e][1]
            obs_numpy_dropped_packets[self.edges_to_id[e]] = unified_obs_dict[e][2]
        self.all_edges_load = unified_dict
        if self.total_counter != 0: #make sure we don't divide by 0
            obs_numpy_array_packets = obs_numpy_array_packets / self.total_counter
            obs_numpy_array_load_link = obs_numpy_array_load_link / 1
            obs_numpy_dropped_packets = obs_numpy_dropped_packets / self.total_counter
        return np.concatenate((obs_numpy_array_packets, obs_numpy_array_load_link, obs_numpy_dropped_packets))

    # mat[i][j] - how much edge i is interfering edge j
    def adopt_interference(self, action_list):
        if action_list is None:
            return
        edges_to_id = self.interference.edges_to_id
        interfered_mat = self.interference.interference_matrix_edges
        updated_actions_list = copy.deepcopy(action_list)
        for interfering_edge in action_list.keys():  # the amount edge is interfering all the other edges.
            interfering_edge_id = edges_to_id[interfering_edge]
            for interfered_edge in updated_actions_list.keys():
                if interfering_edge == interfered_edge:
                    continue
                else:
                    interfered_edge_id = edges_to_id[interfered_edge]
                    interfering_edge_chosen_power = float(action_list[interfering_edge])
                    interference_amount_power = float(interfering_edge_chosen_power * interfered_mat[interfering_edge_id][interfered_edge_id])
                    updated_actions_list[interfered_edge] -= interference_amount_power
                    if updated_actions_list[interfered_edge] < 0:
                        updated_actions_list[interfered_edge] = 0
        return updated_actions_list

    #First zeroes all the links in the system.
    #Second each chosen power of interfernece is adopted by the interference model.
    #Third it updates the effective power of each link and set it accordingly.
    def update_active_links_and_bw(self, action_list=None):
        for name, router in self.routers_list.items():  # update the active links for the next step
            router.clear_links()
            router.zero_bw()
        new_action_list = self.adopt_interference(action_list)
        if action_list is not None:
            for name, router in self.routers_list.items():  # update the active links for the next step
                router.update_active_links(new_action_list)

    #Clear flows (that are made of packets) in all of the routers. Should be done at the start of a step.
    def clear_flows(self):
        for r in self.routers_list:
            self.routers_list[r].remove_all_flows()

    #A compatible reset for StableBaseline3 training.
    #StableBaseline3's reset. Packets should be generated randomly here, and not use a list of pre-ready of interfernce model, packets ready for eval as in custom reset.
    def reset(self):
        self.step_count = 0
        self.total_counter = 0
        self.global_counter = 0
        self.dropped_packets = 0
        self.clear_flows()
        self.update_active_links_and_bw(None)
        self.generate_demand_random()
        for r in self.routers_list:
            self.routers_list[r].get_buffers_observations()
        observation_space = np.concatenate((self.get_state_observation().flatten(), self.interference.interference_matrix_edges.flatten())) #observation - loads + interference
        logger.info(f"Reset took place")
        return observation_space

    # A custom reset method for evaluating RLs algorithm vs a baseline Greedy algorithm.
    # If we want to compare between baseline-greedy and RL, then we should the same interference, demand matrix for the eval, therefore use eval is True.
    def reset_custom(self, episode, eval):
        self.step_count = 0
        self.total_counter = 0
        self.global_counter = 0
        self.dropped_packets = 0
        self.clear_flows()
        if eval is True:
            self.generate_demand_comparison(self.episode_demand_eval, episode)
        else:
            self.generate_demand_comparison(self.episode_demand_train, episode)
        for r in self.routers_list:
            self.routers_list[r].get_buffers_observations()
        self.update_active_links_and_bw(None)
        numpy_obs_unified = np.concatenate((self.get_state_observation().flatten(), self.interference.interference_matrix_edges.flatten()))
        return numpy_obs_unified

    # reward function for RL.
    def reward(self, actions, moving_packets, dropped_packets, total_packets, SF):
        if total_packets > 0:
            reward = -0.01 + moving_packets / total_packets - (dropped_packets / total_packets) * SF
            return reward
        else:
            return 0

    #The step of StableBaseline 3 traing phase.
    #
    def step(self, given_action=None):
        self.step_count += 1
        converted_actions = self.convert_actions_to_edges(given_action)  ##getting which links to open
        self.update_active_links_and_bw(converted_actions)  ##opening the links according to it
        terminated, counter = self.process_flows()  ##moving the flows to the next step, counter is the moving packets from the last step( how many packets were in transit)
        dropped_packets_step = self.get_dropped_packets()  # how many packets were dropped during the step.
        self.dropped_packets += dropped_packets_step
        done_flag = terminated == 1
        numpy_obs = self.get_state_observation()
        numpy_obs_unified = np.concatenate((numpy_obs.flatten(), self.interference.interference_matrix_edges.flatten()))
        reward_value = self.reward(converted_actions, counter, dropped_packets_step, self.total_counter, 10)
        return numpy_obs_unified, reward_value, done_flag, {}

    def convert_actions_to_edges(self, actions_dict):
        id_to_edges_dict = self.interference.id_to_edges
        if isinstance(actions_dict, collections.Mapping):
            return actions_dict
        else:
            translated_actions_dict = {}
            for i in range(0, len(actions_dict)):
                translated_actions_dict[id_to_edges_dict[i]] = actions_dict[i] * 0.01
            return translated_actions_dict

    def render(self, mode='console'):
        pass

    def close(self):
        pass
