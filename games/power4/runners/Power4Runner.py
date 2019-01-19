import os

# import keras.engine.saving import load_model
# from tensorflow import keras
import time

from agents.CommandLineAgent import CommandLineAgent
from agents.RandomAgent import RandomAgent
from agents.TabularQLearningAgent import TabularQLearningAgent
from agents.ReinforceClassicWithMultipleTrajectoriesAgent import ReinforceClassicWithMultipleTrajectoriesAgent

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from environments import Agent
from environments.GameRunner import GameRunner
from environments.GameState import GameState
from games.power4.Power4GameState import Power4GameState
import numpy as np


class Power4Runner(GameRunner):

    def __init__(self, agent1: Agent, agent2: Agent,
                 print_and_reset_score_history_threshold=None,
                 replace_player1_with_commandline_after_similar_results=None):
        self.agents = (agent1, agent2)
        self.stuck_on_same_score = 0
        self.prev_history = None
        self.print_and_reset_score_history_threshold = print_and_reset_score_history_threshold
        self.replace_player1_with_commandline_after_similar_results = replace_player1_with_commandline_after_similar_results

    def run(self, max_rounds: int = -1,
            initial_game_state: Power4GameState = Power4GameState()) -> 'Tuple[float]':
        round_id = 0

        # AJOUT LOG#
        filename = "../logs/" + type(self.agents[0]).__name__ + "_VS_" + type(self.agents[1]).__name__ + str(
            time.time()) + ".txt"
        logs_scores_file = open(filename, "w")
        logs_scores_file.close()
        print(type(self.agents[1]).__name__)
        #####

        score_history = np.array((0, 0, 0))
        while round_id < max_rounds or round_id == -1:
            gs = initial_game_state.copy_game_state()
            terminal = False
            while not terminal:
                current_player = gs.get_current_player_id()
                action_ids = gs.get_available_actions_id_for_player(current_player)
                info_state = gs.get_information_state_for_player(current_player)

                #### AJOUT TIMER
                if round_id == 1000 or round_id == 10000 or round_id == 100000 or round_id == 1000000:
                    timer = time.time()
                ######

                action = self.agents[current_player].act(current_player,
                                                         info_state,
                                                         action_ids)
                ###AJOUT
                if round_id == 1000 or round_id == 10000 or round_id == 100000 or round_id == 1000000:
                    logs_scores_file = open(filename, "a")
                    logs_scores_file.write("TIMER " + str(round_id) + " PLAYER " + str(current_player) + " == " + str(
                        time.time() - timer) + "\n")
                    logs_scores_file.close()
                ####

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
                if round_id == 1000 or round_id == 10000 or round_id == 100000 or round_id == 1000000:
                    logs_scores_file = open(filename, "a")
                    logs_scores_file.write("CHECKPOINT " + str(round_id) + " == " +str(score_history / self.print_and_reset_score_history_threshold) + "\n")
                    logs_scores_file.close()
                if self.print_and_reset_score_history_threshold is not None and \
                        round_id % self.print_and_reset_score_history_threshold == 0:
                    print(score_history / self.print_and_reset_score_history_threshold)
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
                    score_history = np.array((0, 0, 0))
        logs_scores_file.close()
        return tuple(score_history)


if __name__ == "__main__":


    print(Power4Runner(TabularQLearningAgent(), RandomAgent(),
                       print_and_reset_score_history_threshold=100).run(100000000000))
