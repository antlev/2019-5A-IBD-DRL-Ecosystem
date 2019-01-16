import os
from time import time

from agents.TabularQLearningAgent import TabularQLearningAgent

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from agents.RandomAgent import RandomAgent
from environments import Agent
from environments.GameRunner import GameRunner
import tensorflow as tf
from games.windjammers.WindJammersGameState import WindJammersGameState


class TensorboardInstrumentedWindJammersRunner2(GameRunner):

    def __init__(self, agent1: Agent, agent2: Agent, log_dir_root="./logs/" + str(time()), raw_log_dir_root="./logs/" + str(time())):
        self.agents = (agent1, agent2)
        self.writer = tf.summary.FileWriter(log_dir_root)
        self.writer2 = tf.summary.FileWriter(log_dir_root+"_2")
        self.checkpoint = 100

    def run(self, max_rounds: int = -1,
            initial_game_state: WindJammersGameState = WindJammersGameState()):
        episode_id = 1
        agent_1_win=0
        agent_2_win=0
        draw=0
        raw_logs = open("raw_logs" + str(time()), "w+")

        while episode_id < max_rounds or max_rounds == -1:
            gs = initial_game_state.copy_game_state()
            terminal = False
            round_step = 0
            self.mean_action_duration_sum = {0: 0.0, 1: 0.0}
            self.mean_accumulated_reward_sum = {0: 0.0, 1: 0.0}
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

            self.writer.add_summary(tf.Summary(
                value=[
                    tf.Summary.Value(tag="agent1_action_mean_duration",
                                     simple_value=self.mean_action_duration_sum[0] / round_step),

                    tf.Summary.Value(tag="agent2_action_mean_duration",
                                     simple_value=self.mean_action_duration_sum[1] / round_step),

                    tf.Summary.Value(tag="agent1_accumulated_reward",
                                     simple_value=self.mean_accumulated_reward_sum[0]),

                    tf.Summary.Value(tag="agent2_accumulated_reward",
                                     simple_value=self.mean_accumulated_reward_sum[1])

                ],
            ), episode_id)

            if episode_id % self.checkpoint == 0 and episode_id != 0:
                self.writer.add_summary(tf.Summary(
                    value=[
                        tf.Summary.Value(tag=self.agents[0].toString() + "_win",
                                         simple_value=agent_1_win / self.checkpoint),

                        tf.Summary.Value(tag=self.agents[1].toString() + "_win",
                                         simple_value=agent_2_win / self.checkpoint),

                        tf.Summary.Value(tag="draw",
                                         simple_value=draw / self.checkpoint)

                    ],
                ), episode_id)
                raw_logs.write("checkpoint episode " + str(episode_id) + " : " + self.agents[0].toString() + " " + str(agent_1_win) + "% win | " + self.agents[1].toString() + " " + str(agent_2_win) + "% win | draw : " + str(draw) + "%")
                agent_1_win = 0
                agent_2_win = 0
                draw = 0

            episode_id += 1
        raw_logs.close()




if __name__ == "__main__":
    log_dir_root="./logs/Rdm_Vs_TQL" + str(time())
    raw_log_dir_root="./raw_logs/Rdm_Vs_TQL" + str(time())
    print(str(log_dir_root))
    print(TensorboardInstrumentedWindJammersRunner2(RandomAgent(),
                                                   TabularQLearningAgent(),
                                                   log_dir_root=log_dir_root,
                                                   raw_log_dir_root=raw_log_dir_root).run(1000000))
