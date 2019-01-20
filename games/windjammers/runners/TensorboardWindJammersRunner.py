import os
from time import time

import tensorflow as tf

from agents.DeepQLearningAgent import DeepQLearningAgent
from agents.DoubleQLearningAgent import DoubleQLearningAgent
from agents.PPOWithMultipleTrajectoriesMultiOutputsAgent import PPOWithMultipleTrajectoriesMultiOutputsAgent
from agents.ReinforceClassicAgent import ReinforceClassicAgent
from agents.TabularQLearningAgent import TabularQLearningAgent
from agents.RandomAgent import RandomAgent
# RandomRolloutAgent, DeepQLearningAgent
# CommandLineAgent, MOISMCTSWithRandomRolloutsAgent,
# MOISMCTSWithRandomRolloutsExpertThenApprenticeAgent, MOISMCTSWithValueNetworkAgent, PPOWithMultipleTrajectoriesMultiOutputsAgent, \
# ReinforceClassicAgent, ReinforceClassicWithMultipleTrajectoriesAgent\
# ,ReinforceClassicWithMultipleTrajectoriesCustom, ReinforceClassicWithMultipleTrajectoriesCustom2, DoubleDeepQLearningAgent, DoubleQLearningAgent
from environments import Agent
from environments.GameRunner import GameRunner
from games.windjammers.WindJammersGameState import WindJammersGameState

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class TensorboardWindJammersRunner(GameRunner):

    def __init__(self, agent1: Agent, agent2: Agent, log_dir="./logs/" + str(time()), checkpoint=1000):
        self.agents = (agent1, agent2)
        self.writerAgent1 = tf.summary.FileWriter(log_dir + "_" + agent1.toString())
        self.writerAgent2 = tf.summary.FileWriter(log_dir + "_" + agent2.toString())
        self.writerDraw = tf.summary.FileWriter(log_dir + "_" + "Draw")
        tf.summary.merge_all
        self.writerRewardAgent1 = tf.summary.FileWriter(log_dir + "_" + agent1.toString() + "_Reward")
        self.writerRewardAgent2 = tf.summary.FileWriter(log_dir + "_" + agent2.toString() + "_Reward")
        tf.summary.merge_all
        self.writerTimeAgent1 = tf.summary.FileWriter(log_dir + "_" + agent1.toString() + "_Mean_Time")
        self.writerTimeAgent2 = tf.summary.FileWriter(log_dir + "_" + agent2.toString() + "_Mean_Time")
        tf.summary.merge_all
        self.checkpoint = checkpoint
        self.mean_action_duration_sum = {0: 0.0, 1: 0.0}
        self.mean_accumulated_reward_sum = {0: 0.0, 1: 0.0}

    def run(self, max_rounds: int = -1,
            initial_game_state: WindJammersGameState = WindJammersGameState()):
        episode_id = 1
        agent_1_win = 0
        agent_2_win = 0
        draw = 0

        while episode_id < max_rounds or max_rounds == -1:
            gs = initial_game_state.copy_game_state()
            terminal = False
            round_step = 0
            while not terminal:
                # print(gs)
                current_player = gs.get_current_player_id()
                action = 0
                if current_player != -1:
                    action_ids = gs.get_available_actions_id_for_player(current_player)
                    info_state = gs.get_information_state_for_player(current_player)
                    action_time = time()
                    action = self.agents[current_player].act(current_player,
                                                             info_state,
                                                             action_ids)
                    action_time = time() - action_time
                    self.mean_action_duration_sum[current_player] += action_time

                # WARNING : Two Players Zero Sum Game Hypothesis
                (gs, score, terminal) = gs.step(current_player, action)
                self.agents[0].observe(score, terminal)
                self.agents[1].observe(-score, terminal)

                self.mean_accumulated_reward_sum[0] = score
                self.mean_accumulated_reward_sum[1] = -score

            if score > 0:
                agent_1_win += 1
            elif score < 0:
                agent_2_win += 1
            else:
                draw += 1

            round_step += 1

            if episode_id % self.checkpoint == 0 and episode_id != 0:
                self.writerTimeAgent1.add_summary(tf.Summary(
                    value=[tf.Summary.Value(tag="Time", simple_value=self.mean_action_duration_sum[0] / round_step)],
                ), episode_id)
                self.writerTimeAgent1.flush()
                self.writerTimeAgent2.add_summary(tf.Summary(
                    value=[tf.Summary.Value(tag="Time", simple_value=self.mean_action_duration_sum[1] / round_step)],
                ), episode_id)
                self.writerTimeAgent2.flush()

                self.writerRewardAgent1.add_summary(tf.Summary(
                    value=[tf.Summary.Value(tag="Reward", simple_value=self.mean_accumulated_reward_sum[0])],
                ), episode_id)
                self.writerRewardAgent1.flush()
                self.writerRewardAgent2.add_summary(tf.Summary(
                    value=[tf.Summary.Value(tag="Reward", simple_value=self.mean_accumulated_reward_sum[1])],
                ), episode_id)
                self.writerRewardAgent2.flush()
                self.writerAgent1.add_summary(tf.Summary(
                    value=[tf.Summary.Value(tag="Score", simple_value=agent_1_win / self.checkpoint)],
                ), episode_id)
                self.writerAgent1.flush()
                self.writerAgent2.add_summary(tf.Summary(
                    value=[tf.Summary.Value(tag="Score", simple_value=agent_2_win / self.checkpoint)],
                ), episode_id)
                self.writerAgent2.flush()
                self.writerDraw.add_summary(tf.Summary(
                    value=[tf.Summary.Value(tag="Score", simple_value=draw / self.checkpoint)],
                ), episode_id)
                self.writerDraw.flush()
                agent_1_win = 0
                agent_2_win = 0
                draw = 0
                self.mean_action_duration_sum = {0: 0.0, 1: 0.0}
                self.mean_accumulated_reward_sum = {0: 0.0, 1: 0.0}

            episode_id += 1


if __name__ == "__main__":
    # log_dir = "./logs/Rdm_Vs_TQL_2/" + str(time())
    # print(str(log_dir))
    # print(TensorboardWindJammersRunner(RandomAgent(),
    #                                    TabularQLearningAgent(),
    #                                    checkpoint=100,
    #                                    log_dir=log_dir).run(1000000))

    # log_dir = "./logs/Rdm_Vs_Rdm/" + str(time())
    # print(str(log_dir))
    # print(TensorboardWindJammersRunner(RandomAgent(),
    #                                    RandomAgent(),
    #                                    checkpoint=100,
    #                                    log_dir=log_dir).run(1000000))

    # log_dir = "./logs/TQL_Vs_TQL/" + str(time())
    # print(str(log_dir))
    # print(TensorboardWindJammersRunner(TabularQLearningAgent(),
    #                                    TabularQLearningAgent(),
    #                                    checkpoint=100,
    #                                    log_dir=log_dir).run(1000000))

    log_dir = "./logs/Rein_Vs_Rein/" + str(time())
    print(str(log_dir))
    print(TensorboardWindJammersRunner(DoubleQLearningAgent(),
                                       RandomAgent(),
                                       checkpoint=100,
                                       log_dir=log_dir).run(1000000))
