import copy
import time
from datetime import datetime
from functools import reduce
import operator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import plotly.express as px
# from pandas_profiling import ProfileReport

def update_discrete(power_levels_list, power_output, discrete_power_output, update_edge):
    for i in range(1, len(power_levels_list)):
        if power_output[update_edge] < power_levels_list[i]:
            discrete_power_output[update_edge] = power_levels_list[i - 1]
            return

class greedy_baseline():
    def __init__(self, simulated_network, edges_info, shortest_path_list, interference_model, opts, env, id_to_edges,
                 edges_to_id_dict):
        self.name = "greedy_base_line"
        self.simulated_network = simulated_network
        self.edges_info = edges_info
        self.shortest_path_list = shortest_path_list
        self.id_to_edges = id_to_edges
        self.opts = opts
        self.env = env
        self.interference_matrix = interference_model
        self.edges_to_id = edges_to_id_dict

    def find_max_edge(self, given_edges):
        max_edge = 0
        for edge in given_edges.keys():
            max_edge = given_edges[edge][0]
            break
        for edge in given_edges.keys():
            for value in given_edges[edge]:
                if value[2] > max_edge[2]:
                    max_edge = value
        return max_edge

    def interfered_greedy(self, k, obs, power_level):
        edges_capacity = copy.deepcopy(self.env.net.edges_weight)
        sorted_all_edges_capacity_dict = copy.deepcopy(self.env.net.edges_weight)
        interference_matrix = self.env.interference.interference_matrix_edges
        # interference_matrix = np.array([[0, 0.3, 0.5], [0.3, 0, 0.1], [0.4, 0.1, 0]])
        links_status = self.env.get_total_data_per_link() * self.env.total_counter
        for edge in sorted_all_edges_capacity_dict.keys():
            sorted_all_edges_capacity_dict[edge] = min(links_status[0][self.edges_to_id[edge]], edges_capacity[edge])
        possible_edges = {x: x for x in sorted_all_edges_capacity_dict.keys()}
        profit_dict, actions_dict = {}, {}
        import operator
        # init phase, we choose the strongest link, the rest will be accordingly
        maximum_edge = max(sorted_all_edges_capacity_dict.items(), key=operator.itemgetter(1))
        actions_dict[maximum_edge[0]] = 1
        profit_dict[maximum_edge[0]] = (1, (1, maximum_edge[1]))  # dict - key: edge, value - tuple( power, link_status) so we have gain
        # del sorted_all_edges_capacity_dict[maximum_edge[0]]
        del possible_edges[maximum_edge[0]]
        while len(possible_edges) > 0:
            profit_edge_per_round = {}
            for edge in possible_edges.keys():
                profit_edge_per_round[edge] = []
                for level in power_level:
                    effective_power = level
                    for chosen_edge in actions_dict.keys():
                        effective_power -= actions_dict[chosen_edge] * interference_matrix[self.edges_to_id[chosen_edge]][self.edges_to_id[edge]]
                    if effective_power < 0 : effective_power = 0
                    gain = effective_power * sorted_all_edges_capacity_dict[edge]
                    loss = 0
                    for chosen_edge in actions_dict.keys():
                        power_chosen_edge = actions_dict[chosen_edge] - effective_power * interference_matrix[self.edges_to_id[edge]][self.edges_to_id[chosen_edge]] #old edge new power
                        if power_chosen_edge < 0 : power_chosen_edge = 0
                        loss += (profit_dict[chosen_edge][1][0] * profit_dict[chosen_edge][1][1] - power_chosen_edge * profit_dict[chosen_edge][1][1])
                    profit_edge_per_round[edge].append((edge, level, gain - loss))
            most_profitable_edge = self.find_max_edge(profit_edge_per_round)
            if most_profitable_edge[2] < 0:
                actions_dict[most_profitable_edge[0]] = 0
                profit_dict[most_profitable_edge[0]] = (0, (0,most_profitable_edge[2]))
            else:
                actions_dict[most_profitable_edge[0]] = most_profitable_edge[1]
                profit_dict[most_profitable_edge[0]] = (most_profitable_edge[1], (most_profitable_edge[1],most_profitable_edge[2]))
            del possible_edges[most_profitable_edge[0]]
        return actions_dict


    def test_run_greedy_no_interference(self, env, data_avg_list, steps_count_list, done_list, k, function, opts, e):
        counter_steps = 0
        self.env.reset_custom(episode=e, eval=True)
        data_avg_list.append(env.total_counter)
        numpy_obs = env.get_state_observation()
        time_list = []
        while True:
            start = time.time()
            dict_to_insert = function(k, numpy_obs)
            time_list.append(time.time() - start)
            obs_dict, rewards_dict, terminated, info = env.step(dict_to_insert)
            numpy_obs = env.get_state_observation()
            if terminated is True or counter_steps >= opts.max_step_per_episode:
                steps_count_list.append(counter_steps)
                if counter_steps >= opts.max_step_per_episode:
                    done_list.append(0)
                    break
                else:
                    done_list.append(1)
                    break
                break
            counter_steps += 1
        return data_avg_list, steps_count_list, done_list, sum(time_list) / len(time_list)

    def test_run_greedy_with_interference(self, env, data_avg_list, steps_count_list, done_list, data_left, k, function, opts, e,
                                          power_level, actions_episode,dropped_packets):
        counter_steps = 0
        self.env.reset_custom(episode=e, eval=True)
        data_avg_list.append(env.total_counter)
        numpy_obs = env.get_state_observation()
        time_list = []
        actions_per_episode = []
        while True:
            start = time.time()
            dict_to_insert = function(k, numpy_obs, [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1])
            time_list.append(time.time() - start)
            obs_dict, rewards_dict, terminated, info = env.step(dict_to_insert)
            numpy_obs = env.get_state_observation()
            if terminated is True or counter_steps >= opts.max_step_per_episode:
                steps_count_list.append(counter_steps)
                if counter_steps >= opts.max_step_per_episode:
                    done_list.append(0)
                    data_left.append(env.total_counter)
                    dropped_packets.append(env.dropped_packets)
                    break
                else:
                    actions_episode.append(actions_per_episode)
                    done_list.append(1)
                    data_left.append(env.total_counter)
                    dropped_packets.append(env.dropped_packets)
                    break
                break
            counter_steps += 1
        return data_avg_list, steps_count_list, done_list, sum(time_list) / len(time_list), data_left, actions_episode, dropped_packets

    def greedy_baseline_method(self, opts):
        data_list_with_inter, steps_list_with_inter, done_list_with_inter, avg_time_with_inter, data_left_with_inter,dropped_packets = [],[], [], [], [], []
        actions_episode = []
        total_power_per_interference = []
        for e in range(0, opts.episode_eval):
            data_list_with_inter, steps_list_with_inter, done_list_with_inter, avg_time1, data_left_with_inter, actions_episode,dropped_packets = \
                self.test_run_greedy_with_interference(self.env, data_list_with_inter, steps_list_with_inter,done_list_with_inter, data_left_with_inter, 1, self.interfered_greedy,
                                                       opts, e, 5, actions_episode,dropped_packets=dropped_packets)
            avg_time_with_inter.append(float(avg_time1))
            print(str(e) + " DONE")
            if e % 10 == 0:
                calc_total_power = []
                calc_total_power.append(np.array([df.values for df in actions_episode[e]]).squeeze().sum())
                total_power_per_interference.append(sum(calc_total_power) / len(calc_total_power))

        name_for_file = str(self.opts.outputname) + "_" + str(datetime.now().strftime("%d-%m-%Y,%H:%M:%S")) + "-greedy" + "-scene:" + str(self.opts.run_scene) + "-num:" + str(opts.episode_eval) + ".txt"
        config_for_test = str(opts)
        average_greedy, dropped_packets_avg = [], []
        for i in range(int (opts.episode_eval/10)):
            average_greedy.append(float(sum(steps_list_with_inter[int(i * (opts.episode_eval / 19)):int((i+1) * (opts.episode_eval / 19))]) / len(
                steps_list_with_inter[int(i * (opts.episode_eval / 19)):int((i+1) * (opts.episode_eval / 19))])))
            dropped_packets_avg.append(float(sum(dropped_packets[int(i * (opts.episode_eval / 19)):int((i + 1) * (opts.episode_eval / 19))]) / len(
                dropped_packets[int(i * (opts.episode_eval / 19)):int((i + 1) * (opts.episode_eval / 19))])))
        file_to_write = open(name_for_file, "w")
        concat_str = ""
        for i in range(int(opts.episode_eval/10)):
            concat_str += "[" + str(i * (opts.episode_eval / 19)) + "," + str(int(i+1) * (opts.episode_eval / 19)) + "] interference level is uniform(0," + str(opts.eval_numbers[i]) + ") interfered greedy avg is : " + str(average_greedy[i]) + " total average power is: "+str(total_power_per_interference[i])+\
                          " dropped packets is : " + str(dropped_packets_avg[i]) + "\n"
        L = [
            config_for_test + "\n\n",
            "\n\nGREEDY WITH INTERFERENCE: \n",
            "data list is:" + str(data_list_with_inter) + "\n",
            "steps list is:" + str(steps_list_with_inter) + "\n",
            "done list is:" + str(done_list_with_inter) + "\n",
            "left data list is:" + str(data_left_with_inter) + "\n",
            "Average time is for decision " + str((sum(avg_time_with_inter) / len(avg_time_with_inter))) + " seconds.\n\n\n",
            concat_str
        ]
        file_to_write.writelines(L)
        file_to_write.close()
        return data_list_with_inter, steps_list_with_inter, done_list_with_inter, data_left_with_inter
