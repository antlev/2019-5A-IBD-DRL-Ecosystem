import os

from agents.CommandLineAgent import CommandLineAgent
from agents.DeepQLearningAgent import DeepQLearningAgent
from agents.DoubleDeepQLearning import DoubleDeepQLearningAgent
from agents.DoubleQLearningAgent import DoubleQLearningAgent
from agents.MOISMCTSWithRandomRolloutsAgent import MOISMCTSWithRandomRolloutsAgent
from agents.MOISMCTSWithRandomRolloutsExpertThenApprenticeAgent import \
    MOISMCTSWithRandomRolloutsExpertThenApprenticeAgent
from agents.MOISMCTSWithValueNetworkAgent import MOISMCTSWithValueNetworkAgent
from agents.PPOWithMultipleTrajectoriesMultiOutputsAgent import PPOWithMultipleTrajectoriesMultiOutputsAgent
from agents.RandomAgent import RandomAgent
from agents.RandomRolloutAgent import RandomRolloutAgent
from agents.ReinforceClassicAgent import ReinforceClassicAgent
from agents.ReinforceClassicWithMultipleTrajectoriesAgent import ReinforceClassicWithMultipleTrajectoriesAgent
from agents.TabularQLearningAgent import TabularQLearningAgent
from games.tictactoe.runners.SafeTicTacToeRunner import SafeTicTacToeRunner

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from time import time, sleep
import tensorflow as tf
from environments import Agent
from environments.GameRunner import GameRunner
from games.power4.Power4GameState import Power4GameState
import numpy as np

class TensorboardPower4Runner(GameRunner):

    def __init__(self, agent1: Agent, agent2: Agent,
                 log_and_reset_score_history_threshold=None,
                 replace_player1_with_commandline_after_similar_results=None,
                 log_dir="./logs/" + str(time())):
        self.agents = (agent1, agent2)
        self.stuck_on_same_score = 0
        self.prev_history = None
        self.log_and_reset_score_history_threshold = log_and_reset_score_history_threshold
        self.replace_player1_with_commandline_after_similar_results = replace_player1_with_commandline_after_similar_results
        self.writerAgent1 = tf.summary.FileWriter(log_dir + "_" + agent1.toString() + "_Agent1")
        self.writerAgent2 = tf.summary.FileWriter(log_dir + "_" + agent2.toString() + "_Agent2")
        self.writerDraw = tf.summary.FileWriter(log_dir + "_" + "Draw")
        tf.summary.merge_all
        self.writerTimeAgent1 = tf.summary.FileWriter(log_dir + "_" + agent1.toString() + "_Mean_Time")
        self.writerTimeAgent2 = tf.summary.FileWriter(log_dir + "_" + agent2.toString() + "_Mean_Time")
        tf.summary.merge_all
        self.mean_action_duration_sum = {0: 0.0, 1: 0.0}
        self.mean_accumulated_reward_sum = {0: 0.0, 1: 0.0}

    def run(self, max_rounds: int = -1,
            initial_game_state: Power4GameState = Power4GameState()) -> 'Tuple[float]':
        round_id = 0

        time()
        score_history = np.array((0, 0, 0))
        while round_id < max_rounds or round_id == -1:
            if round_id == 0:
                gs = initial_game_state.copy_game_state()
            else:
                gs.newGameState()
            terminal = False
            while not terminal:
                # print(gs)
                # sleep(0.1)
                current_player = gs.get_current_player_id()
                action_ids = gs.get_available_actions_id_for_player(current_player)
                info_state = gs.get_information_state_for_player(current_player)
                action_time = time()
                action = self.agents[current_player].act(current_player,
                                                         info_state,
                                                         action_ids)
                self.mean_action_duration_sum[current_player] += time() - action_time

                # WARNING : Two Players Zero Sum Game Hypothesis
                (gs, score, terminal) = gs.step(current_player, action)
                self.agents[current_player].observe(
                    (1 if current_player == 0 else -1) * score,
                    terminal)

                if terminal:
                    score_history += (1 if score == 1 else 0, 1 if score == -1 else 0, 1 if score == 0 else 0)
                    other_player = (current_player + 1) % 2
                    self.agents[other_player].observe(
                        (1 if other_player == 0 else -1) * score,
                        terminal)

            if round_id != -1:
                round_id += 1
                if self.log_and_reset_score_history_threshold is not None and \
                        round_id % self.log_and_reset_score_history_threshold == 0:
                    print(score_history / self.log_and_reset_score_history_threshold)
                    if self.prev_history is not None and \
                            score_history[0] == self.prev_history[0] and \
                            score_history[1] == self.prev_history[1] and \
                            score_history[2] == self.prev_history[2]:
                        self.stuck_on_same_score += 1
                    else:
                        self.prev_history = score_history
                        self.stuck_on_same_score = 0
                    if (self.replace_player1_with_commandline_after_similar_results is not None and
                            self.stuck_on_same_score >= self.replace_player1_with_commandline_after_similar_results):
                        self.agents = (CommandLineAgent(), self.agents[1])
                        self.stuck_on_same_score = 0

                    self.writerTimeAgent1.add_summary(tf.Summary(
                        value=[tf.Summary.Value(tag="Time",
                                                simple_value=self.mean_action_duration_sum[0] / round_id)],
                    ), round_id)
                    self.writerTimeAgent1.flush()
                    self.writerTimeAgent2.add_summary(tf.Summary(
                        value=[tf.Summary.Value(tag="Time",
                                                simple_value=self.mean_action_duration_sum[1] / round_id)],
                    ), round_id)
                    self.writerTimeAgent2.flush()
                    self.writerAgent1.add_summary(tf.Summary(
                        value=[tf.Summary.Value(tag="Score", simple_value=score_history[0] / self.log_and_reset_score_history_threshold)],
                    ), round_id)
                    self.writerAgent1.flush()
                    self.writerAgent2.add_summary(tf.Summary(
                        value=[tf.Summary.Value(tag="Score", simple_value=score_history[1] / self.log_and_reset_score_history_threshold)],
                    ), round_id)
                    self.writerAgent2.flush()
                    self.writerDraw.add_summary(tf.Summary(
                        value=[tf.Summary.Value(tag="Score", simple_value=score_history[2] / self.log_and_reset_score_history_threshold)],
                    ), round_id)
                    self.writerDraw.flush()

                    self.mean_action_duration_sum = {0: 0.0, 1: 0.0}
                    self.mean_accumulated_reward_sum = {0: 0.0, 1: 0.0}
                    score_history = np.array((0, 0, 0))
        return tuple(score_history)


if __name__ == "__main__":


    log_dir = "./logs/Random_VS_TabularQLearning_" + str(time())
    print(log_dir)
    print(TensorboardPower4Runner(RandomAgent(),
                                     TabularQLearningAgent(),
                                     log_and_reset_score_history_threshold=1000,
                                     log_dir=log_dir).run(1000000000))

    # AGENTS EXAMPLES :
    # CommandLineAgent()
    # RandomAgent()
    # RandomRolloutAgent(3, SafeTicTacToeRunner(RandomAgent(), RandomAgent()))
    # TabularQLearningAgent()
    # DeepQLearningAgent(9,9)
    # ReinforceClassicAgent(9,9)
    # ReinforceClassicWithMultipleTrajectoriesAgent(9,9)
    # PPOWithMultipleTrajectoriesMultiOutputsAgent(9,9)
    # MOISMCTSWithRandomRolloutsAgent(100, SafeTicTacToeRunner(RandomAgent(), RandomAgent()))
    # MOISMCTSWithRandomRolloutsExpertThenApprenticeAgent(100, SafeTicTacToeRunner(RandomAgent(), RandomAgent()),9,9)
    # MOISMCTSWithValueNetworkAgent(100, SafeTicTacToeRunner(RandomAgent(), RandomAgent()))

