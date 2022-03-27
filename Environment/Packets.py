import copy

import networkx as nx
from loguru import logger


class Packets:
    def __init__(self, shortest_path_list, given_flow_id=None, source=0, destination=1, size=0.1, name='1'):
        self.name = name
        self.source = source
        self.destination = destination
        self.path_done = [source]
        self.current_location = source
        self.shortest_path_list = copy.copy(shortest_path_list[source][destination])
        if len(self.shortest_path_list) > 1:
            self.next_hop = copy.copy(self.shortest_path_list[1])
        elif len(self.shortest_path_list) == 0:
            return
        else:
            self.next_hop = copy.copy(self.shortest_path_list[0])
        self.flow_id = given_flow_id
        self.size = size

    def get_next_hop(self):
        return self.next_hop

    def packet_step(self):  # return return value, size finished
        self.path_done.append(self.next_hop)
        if self.next_hop != self.current_location:  # on the way
            self.current_location = self.next_hop
            self.shortest_path_list.pop(0)  # chopping from the path
            if self.current_location == self.destination:
                return 1, self.size
            if len(self.shortest_path_list) > 1:
                self.next_hop = self.shortest_path_list[1]  # update next hop
            return 0, 0
