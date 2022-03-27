import copy
import random

import numpy as np
from loguru import logger
from Environment.Packets import Packets
import Misc_Folder.Constants as c
from Environment.Buffers import Buffers
import math


class Stations:
    def __init__(self, name, network_topology=None, shortest_path_list=None, edges_dict=None, transceiver=None,queue_size = None):
        self.name = name
        self.connection_matrix = network_topology
        self.shortest_paths_list = copy.copy(shortest_path_list)
        self.out_links = {}
        self.initialize_out_queues(queue_size)
        self.activated_links_per_step = {}
        for i in self.out_links:
            self.activated_links_per_step[str(self.name) + "_" + str(int(self.out_links[i].out_going_link_to))] = float(0)
        self.max_transceiver = transceiver
        self.current_transceiver = 0

    def initialize_out_queues(self,queue_size):
        direct_links = [val for key, val in self.shortest_paths_list.items()]
        out_going_links = []
        for i in range(len(direct_links)):
            if len(direct_links[i]) == 2:
                out_going_links.insert(0, direct_links[i][1])
        for i in range(len(out_going_links)):  # insert the queues
            self.out_links[out_going_links[i]] = Buffers(self.name, out_going_links[i],
                                                         self.connection_matrix.graph_topology.adj[
                                                                      int(self.name)], queue_size)

    def add_flow(self, flow):
        return self.out_links[flow.next_hop].add_flow_to_q(flow)

    def is_link_activated(self, next_hop):
        return self.activated_links_per_step[next_hop]

    def remove_flow(self, flow, total_count):
        return self.out_links[int(flow.next_hop)].remove_flow_from_q(flow.flow_id,total_counter=total_count)

    def get_buffers_observations(self):
        send_info = {}  # key is destination, value is amount of data
        for q in self.out_links:
            send_info[self.name + "_" + str(self.out_links[q].get_out_going_link())] = \
                (self.out_links[q].get_total_data(), self.out_links[q].get_load_status(), self.out_links[q].get_dropped_packets_in_q())
        self.observation_moving_dropped_packets_prop = send_info

    def get_router_queues_load(self):
        return self.name, self.observation_moving_dropped_packets_prop

    def update_active_links(self, activated_links):  # should get as a dict
        router_self_dict = {}
        for queue, power in activated_links.items():
            if queue.split("_")[0] == self.name:
                router_self_dict[queue] = power
        router_self_dict_sorted = sorted(router_self_dict.items(), key=lambda x: x[1], reverse=True)
        for name, status in router_self_dict_sorted:
            if name in activated_links.keys():
                if math.ceil(activated_links[name]) == 1 and math.floor(self.activated_links_per_step[name]) == 0:
                    self.current_transceiver += 1
                    if self.current_transceiver <= self.max_transceiver:
                        self.activated_links_per_step[name] = copy.deepcopy(float(activated_links[name]))
                        self.out_links[int(name.split("_")[1])].power = copy.deepcopy(float(activated_links[name]))

    def get_all_flows(self):
        flows = []
        for q in self.out_links:
            flows.append(list(self.out_links[q].flows.values()))
        return [j for i in flows for j in i]

    def zero_bw(self):
        for key, link in self.out_links.items():
            link.zero_bw_in_buffer()

    def remove_all_flows(self):
        for q in self.out_links:
            self.out_links[q].remove_all_flow_from_q()

    def clear_links(self):
        self.current_transceiver = 0
        for key, value in self.activated_links_per_step.items():
            if self.activated_links_per_step[key] > 0:
                self.activated_links_per_step[key] = copy.deepcopy(float(0))
                self.out_links[int(key.split("_")[1])].power = copy.deepcopy(float(0))

    def update_chosen_action(self, action):
        self.last_chosen_action = action

    def get_last_chosen_action(self):
        return self.last_chosen_action

    def get_dropped_packets_in_router(self):
        dropped_packets = 0
        for link in self.out_links.keys():
            dropped_packets += self.out_links[link].get_dropped_packets_in_q()
        return dropped_packets