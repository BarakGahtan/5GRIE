import collections
import copy
import math
from Environment.Packets import Packets


class Buffers:
    def __init__(self, source, destination, capacity_dict, max_packets):
        self.source_name = source
        self.out_going_link_to = destination
        self.flows = {}
        self.total_flows_in_q = 0
        self.link_max_capacity = capacity_dict[self.out_going_link_to]['weight']
        self.used_bw = 0
        self.power = float(0)
        self.used_step = 0
        self.current_packets = 0
        self.max_packets = max_packets
        self.dropped_packets = 0

    def remove_flow_from_q(self, f, total_counter):
        original_size = self.flows[f].size
        delta_available_in_q = math.floor(self.link_max_capacity * self.power) - self.used_bw
        if original_size <= delta_available_in_q:  # no problem, whole flow moves
            self.used_bw += original_size
            isFinished, amount = self.flows[f].packet_step()
            if isFinished:
                self.total_flows_in_q -= 1
                self.current_packets -= amount
                del self.flows[f]
                return 1, amount
            else:
                temp_flow = copy.deepcopy(self.flows[f])
                del self.flows[f]
                self.total_flows_in_q -= 1
                self.current_packets -= temp_flow.size
                return 0, temp_flow

        elif 0 < delta_available_in_q < original_size:  # split the flow
            partial_flow = copy.deepcopy(self.flows[f])
            partial_flow.size = delta_available_in_q
            self.flows[f].size -= delta_available_in_q
            self.used_bw += delta_available_in_q
            isFinished, amount = partial_flow.packet_step()
            if isFinished:
                del partial_flow
                self.current_packets -= amount
                return 1, amount
            else:
                self.current_packets -= partial_flow.size
                return 0, partial_flow
        else:  # no bandwidth to move
            return -1, -1

    def add_flow_to_q(self, f: Packets):
        return_amount = 0
        if self.current_packets <= self.max_packets:  #we have packets to add
            delta_to_insert = self.max_packets - self.current_packets # check the delta to add
            if 0 < f.size <= delta_to_insert:
                if f.flow_id in self.flows:
                    self.flows[f.flow_id].size += f.size
                else:
                    self.flows[f.flow_id] = copy.deepcopy(f)
                    self.total_flows_in_q += 1
                self.current_packets += f.size
            elif f.size > delta_to_insert > 0:
                self.dropped_packets += f.size - delta_to_insert
                return_amount += f.size - delta_to_insert
                f.size = delta_to_insert
                if f.flow_id in self.flows:
                    self.flows[f.flow_id].size += f.size
                else:
                    self.flows[f.flow_id] = copy.deepcopy(f)
                    self.total_flows_in_q += 1
                self.current_packets += f.size
            elif delta_to_insert == 0:
                self.dropped_packets += f.size
                return_amount += f.size
        else:
            self.dropped_packets += f.size
            return_amount += f.size
        return return_amount

    def get_out_going_link(self):
        return int(self.out_going_link_to)

    def get_total_data(self):
        total_data = 0
        for f in self.flows:
            total_data += self.flows[f].size
        return total_data

    def zero_bw_in_buffer(self):
        self.used_bw = 0
        self.used_step = 0
        self.dropped_packets = 0

    def remove_all_flow_from_q(self):
        self.flows.clear()

    def get_dropped_packets_in_q(self):
        return self.dropped_packets

    def get_load_status(self):
        return float(self.current_packets / self.max_packets)