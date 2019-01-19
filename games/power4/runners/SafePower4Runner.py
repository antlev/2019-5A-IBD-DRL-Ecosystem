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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from time import time, sleep
import tensorflow as tf
from environments import Agent
from environments.GameRunner import GameRunner
from games.power4.Power4GameState import Power4GameState
import numpy as np

class SafePower4Runner(GameRunner):

    def __init__(self, agent1: Agent, agent2: Agent,
                 log_and_reset_score_history_threshold=None,
                 replace_player1_with_commandline_after_similar_results=None,
                 log_dir="./logs/" + str(time())):
        self.agents = (agent1, agent2)


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
                current_player = gs.get_current_player_id()
                action_ids = gs.get_available_actions_id_for_player()
                info_state = gs.get_information_state_for_player(current_player)
                action = self.agents[current_player].act(current_player,
                                                         info_state,
                                                         action_ids)

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

        return tuple(score_history)