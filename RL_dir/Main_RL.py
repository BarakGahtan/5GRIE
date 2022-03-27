import os
import statistics
import time
from datetime import datetime

import numpy as np
import torch
from colorama import Fore
from loguru import logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

def init_hidden_layers_rl(input_nn):
    if input_nn == 10:
        return [1024, 1024]


class RunAgents:
    def __init__(self, environment, hp, opts, greedy_steps, steps_greedy_list, train_single_agent, left_data, dropped_packets):
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        self.environment = environment
        self.HP = hp
        self.opts = opts
        self.steps_greedy_list = steps_greedy_list
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.center_flag = self.opts.center_agent == 1
        if self.center_flag is True:
            self.stable_agent(greedy_steps, train_single_agent, self.opts.model_name, left_data, dropped_packets)
        logger.info(f"Finished init Agent Trainers.")

    def stable_agent(self, greedy_steps, train_mode, model_name, left_data, dropped_packets):
        l_r = self.opts.learning_rate  # 3e-5
        given_batch_size = self.opts.batch_size  # 128
        low_interference = self.opts.low_interference
        high_intf = self.opts.high_interference
        name_for_checkpoints = model_name + str(datetime.now().strftime("%d-%m-%Y,%H:%M:%S")) + "_LR" + str(l_r) + "LI_" + str(low_interference) + "_HI_" + str(
            high_intf) + "_checkpoints/"
        checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=name_for_checkpoints)
        x = dict(pi=init_hidden_layers_rl(self.opts.nn_layer), vf=init_hidden_layers_rl(self.opts.nn_layer)) #Two hidden layers.
        policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[x])
        log_dir = "./logdir"
        env = Monitor(self.environment, log_dir)
        if train_mode == 1:
            self.center_model = PPO('MlpPolicy', env, tensorboard_log=log_dir, device='cpu', verbose=1, learning_rate=3e-4, policy_kwargs=policy_kwargs,batch_size=given_batch_size)        #     self.center_model_0.save(model_name)
            self.center_model.learn(total_timesteps=self.opts.times_step, callback=[checkpoint_callback])
            self.center_model.save(model_name)
        else:
            self.center_model_1 = PPO.load("1111", env= env, device = 'cpu')
        parameters_dict = {'net': x, 'learning_rate': l_r, 'batch_size': given_batch_size}
        avg_steps, avg_reward = self.evaluate(greedy_steps, parameters_dict, True, left_data, dropped_packets) #Prints

    def evaluate(self, greedy_steps, params, to_print, left_data, dropped_packets_greedy):
        steps_list, total_epoch_reward_list, data_list, time_list, left_data_rl, dropped_packets = [], [], [], [], [], []
        actions_episode = []
        for i in range(0, self.opts.episode_eval):
            current_state = self.environment.reset_custom(episode=i, eval=True)
            data_list.append(self.environment.total_counter)
            total_epoch_reward, steps, decay_steps = 0, 0, 0
            terminated_flag = False
            actions_per_episode = []
            while not terminated_flag:
                start = time.time()
                actions, _ = self.center_model_1.predict(current_state)
                time_list.append(time.time() - start)
                actions_dict = self.environment.convert_actions_to_edges(actions)
                next_state, rewards_dict, terminated_flag, info = self.take_actions(actions_dict)
                total_epoch_reward += rewards_dict
                current_state = next_state
                if terminated_flag:
                    steps_list.append(steps)
                    total_epoch_reward_list.append(total_epoch_reward)
                    left_data_rl.append(self.environment.total_counter)
                    actions_episode.append(actions_per_episode)
                    dropped_packets.append(self.environment.dropped_packets)
                steps += 1
            logger.info(f"Evaluate episode number " + str(i) + " is finished")
        if total_epoch_reward_list != 0:
            avg_reward = str(sum(total_epoch_reward_list) / len(total_epoch_reward_list))
        else:
            avg_reward = 0
        if total_epoch_reward_list != 0:
            avg_steps = str(sum(steps_list) / len(steps_list))
        else:
            avg_steps = 0
        if to_print:
            name_for_file = str(self.opts.outputname) + "_" + str(datetime.now().strftime("%d-%m-%Y,%H:%M:%S")) + "_" + str(self.opts.model_name) + "_scene_" + str(
                self.opts.run_scene) + "_evaluation_RL.txt"
            config_for_test = str(self.opts)
            file = open(name_for_file, "w")
            average_greedy_with, avg_dropped_packets, average_greedy_rl = [], [], []
            for i in range(int(self.opts.episode_eval/10)):
                average_greedy_with.append(float(sum(self.steps_greedy_list[int(i * (self.opts.episode_eval / 19)):int((i+1) * (self.opts.episode_eval / 19))]) / len(
                self.steps_greedy_list[int(i * (self.opts.episode_eval / 19)):int((i+1) * (self.opts.episode_eval / 19))])))
                avg_dropped_packets.append(float(sum(dropped_packets[int(i * (self.opts.episode_eval / 19)):int((i + 1) * (self.opts.episode_eval / 19))]) / len(
                    dropped_packets[int(i * (self.opts.episode_eval / 19)):int((i + 1) * (self.opts.episode_eval / 19))])))
                average_greedy_rl.append(float(sum(steps_list[int((i) * (self.opts.episode_eval / 19)):int((i + 1) * (self.opts.episode_eval / 19))]) / len(
                    steps_list[int(i * (self.opts.episode_eval / 19)):int((i + 1) * (self.opts.episode_eval / 19))])))
            concat_str = ""
            for i in range(int(self.opts.episode_eval/10)):
                concat_str += "[" + str(i * (self.opts.episode_eval / 19)) + "," + str((i+1) * (self.opts.episode_eval / 19)) + "] interference level is uniform(0," + str(
                    self.opts.eval_numbers[i]) + ")  greedy avg is: " + str(average_greedy_with[i]) + " RL is " + str(average_greedy_rl[i]) + " fraction is greedy/rl " + str((average_greedy_with[i] / float(average_greedy_rl[i])) * 100) + "%" + " dropped packets RL is: " + str(avg_dropped_packets[i]) + " dropped packets greedy is: " + str(dropped_packets_greedy[i]) + "\n"

            lines = [
                config_for_test + str(params) + "\n\n",
                "data list is: " + str(data_list) + "\n",
                "steps list is:" + str(steps_list) + "\n",
                "reward list is:" + str(total_epoch_reward_list) + "\n",
                "left data list is:" + str(left_data_rl) + "\n",
                "Average steps to finish demand matrix is " + avg_steps + "\n",
                "Average reward is " + avg_reward + "\n",
                "Average decision time per slot is " + str(sum(time_list) / len(time_list)) + "\n",
                concat_str
            ]
            file.writelines(lines)
            file.close()
        return avg_steps, avg_reward
