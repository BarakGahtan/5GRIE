import os
from datetime import datetime

import numpy as np
import pandas as pd


def output_calculate_wanted_steps(env):
    minimum_steps_matrix = np.zeros(shape=(len(env.routers_list), len(env.routers_list)))
    reward_matrix = np.zeros(shape=(len(env.routers_list), len(env.routers_list)))
    for i in range(env.next_episode_demand_matrix.shape[0]):
        for j in range(env.next_episode_demand_matrix.shape[1]):
            if i == j:
                continue
            mean_steps = 0
            reward_matrix_number = 0
            flag = 0
            for k in env.global_all_shortest_path[i][j]:
                if env.global_all_shortest_path[i][j][-1:][0] == k: continue
                mean_steps += np.ceil(
                    (float((env.routers_list[str(k)].max_transceiver) / (len(env.routers_list[str(k)].out_links))) *
                     ((env.next_episode_demand_matrix[i][j]) / (env.capacity))))
                # expected mean = probability to choose the edge  * the number of times we need to  use the edge == ceil((transcivers / queues) * (packets/ capacity))
                if flag == 0:
                    reward_matrix_number = np.ceil((float(
                        (env.routers_list[str(k)].max_transceiver) / (len(env.routers_list[str(k)].out_links))) *
                                                    ((env.next_episode_demand_matrix[i][j]) / (env.capacity))))
                    flag = 1
            minimum_steps_matrix[i][j] = mean_steps
            reward_matrix[i][j] = reward_matrix_number

    df = pd.DataFrame(minimum_steps_matrix)
    df.columns = list(env.routers_list.keys())  # Change the 0,1,2,...,9 in the first line into A,B,C,...,J
    df.index = list(env.routers_list.keys())
    df_r = pd.DataFrame(reward_matrix)
    df_r.columns = list(env.routers_list.keys())  # Change the 0,1,2,...,9 in the first line into A,B,C,...,J
    df_r.index = list(env.routers_list.keys())
    exp_dir = '{}/{}/'.format("excel_files", datetime.now().strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    file_name = 'minimum_steps'
    writer = pd.ExcelWriter(exp_dir + file_name + '.xlsx')
    df.to_excel(writer, 'page_1', float_format='%.5f')
    writer.save()
    file_name = 'reward_steps'
    writer_1 = pd.ExcelWriter(exp_dir + file_name + '.xlsx')
    df_r.to_excel(writer_1, 'page_1', float_format='%.5f')
    writer_1.save()


def sum_rewards_per_agent(dict_step, dict_reward):
    for i in dict_step.keys():
        dict_reward[i] += dict_step[i]
    return dict_reward


def output_result_custom_loop(data, steps, done, opts, str_eval, done_training, best_avg_steps, steps_greedy,
                              left_data):
    batch_param = str(opts.batch_size)
    lr_param = str(opts.learning_rate)
    nn_param = str(opts.nn_layer)
    if steps_greedy != 0:
        frac = str(float(best_avg_steps) / float(steps_greedy))
    else:
        frac = 0
    prefix = str(datetime.now()) + "_batch_" + batch_param + "_lr_" + lr_param + "_nn_param" + nn_param
    name_for_file = prefix + "_" + str(str_eval) + "_RL.txt"
    config_for_test = str(opts)
    file1 = open(name_for_file, "w")
    lines = [
        config_for_test + "\n\n",
        "Left data from system : \n" + str(left_data) + "\n",
        "RL System - epsilon greedy: \n" + str(data) + "\n" + "30% system greedy steps count: \n" + str(
            steps) + "\n" + "1 transceivers system done count: \n" + str(done) + "\n",
        "RL System - epsilon greedy system: Average steps to finish demand matrix of an average " + str(
            sum(data) / len(data)) + " packets is " + str(
            sum(steps) / len(steps)) + " total done % " + str((sum(done) / len(done)) * 100) + "\n",
        "RL System - last 20 episodes avg: " + str(
            sum(data[-20:]) / len(data[-20:])) + " packets is " + str(
            sum(steps[-20:]) / len(steps[-20:])) + "steps, total done % " + str(
            (sum(done[-20:]) / len(done[-20:])) * 100) + "\n",
        "Done training after " + str(done_training) + " \n",
        "Best average steps RL " + str(best_avg_steps) + "\n",
        "Best average steps greedy " + str(steps_greedy) + "\n"
                                                           "Fraction is " + str(frac) + "\n"

    ]
    file1.writelines(lines)
    file1.close()


def sum_dict(given_dict):
    sum_ = 0
    for i in range(len(given_dict)):
        sum_ += given_dict[i][1]
    return sum_


def sum_reward_(given_dict):
    value = 0
    for key in given_dict.keys():
        value += float(given_dict[key])
    return value