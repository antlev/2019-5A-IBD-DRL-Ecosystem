import os

from agents.CommandLineAgent import CommandLineAgent
from agents.DeepQLearningAgent import DeepQLearningAgent
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
from time import time
import tensorflow as tf
from environments import Agent
from environments.GameRunner import GameRunner
from games.tictactoe.TicTacToeGameState import TicTacToeGameState
from games.tictactoe.runners.TensorboardTicTacToeRunner import TensorboardTicTacToeRunner

import random
import sys
from threading import Thread
import time

if __name__ == "__main__":
    class Bench(Thread):
        def __init__(self, log_dir_root, opponent):
            Thread.__init__(self)
            self.opponent = opponent
            self.log_dir_root = log_dir_root
            self.time = str(time.time())

        def run(self):
            if self.opponent == "RandomAgent":
                log_dir1 = self.log_dir_root + "TabularQLearningAgent_VS_RandomAgent_" + self.time
                print(log_dir1)
                print(TensorboardTicTacToeRunner(TabularQLearningAgent(),
                                                 RandomAgent(),
                                                 log_and_reset_score_history_threshold=10000,
                                                 log_dir=log_dir1).run(100000000))
            elif self.opponent == "TabularQLearningAgent":
                log_dir2 = self.log_dir_root + "TabularQLearningAgent_VS_TabularQLearningAgent_" + self.time
                print(log_dir2)
                print(TensorboardTicTacToeRunner(TabularQLearningAgent(),
                                                 TabularQLearningAgent(),
                                                 log_and_reset_score_history_threshold=10000,
                                                 log_dir=log_dir2).run(100000000))
            elif self.opponent == "DeepQLearningAgent":
                log_dir3 = self.log_dir_root + "TabularQLearningAgent_VS_DeepQLearningAgent_" + self.time
                print(log_dir3)
                print(TensorboardTicTacToeRunner(TabularQLearningAgent(),
                                                 DeepQLearningAgent(9, 9),
                                                 log_and_reset_score_history_threshold=10000,
                                                 log_dir=log_dir3).run(100000000))
            elif self.opponent == "ReinforceClassicAgent":
                log_dir4 = self.log_dir_root + "TabularQLearningAgent_VS_ReinforceClassicAgent_" + self.time
                print(log_dir4)
                print(TensorboardTicTacToeRunner(TabularQLearningAgent(),
                                                 ReinforceClassicAgent(9, 9),
                                                 log_and_reset_score_history_threshold=10000,
                                                 log_dir=log_dir4).run(100000000))
            elif self.opponent == "ReinforceClassicWithMultipleTrajectoriesAgent":
                log_dir5 = self.log_dir_root + "TabularQLearningAgent_VS_ReinforceClassicWithMultipleTrajectoriesAgent_" + self.time
                print(log_dir5)
                print(TensorboardTicTacToeRunner(TabularQLearningAgent(),
                                                 ReinforceClassicWithMultipleTrajectoriesAgent(9, 9),
                                                 log_and_reset_score_history_threshold=10000,
                                                 log_dir=log_dir5).run(100000000))
            elif self.opponent == "PPOWithMultipleTrajectoriesMultiOutputsAgent":
                log_dir6 = self.log_dir_root + "TabularQLearningAgent_VS_PPOWithMultipleTrajectoriesMultiOutputsAgent_" + self.time
                print(log_dir6)
                print(TensorboardTicTacToeRunner(TabularQLearningAgent(),
                                                 PPOWithMultipleTrajectoriesMultiOutputsAgent(9, 9),
                                                 log_and_reset_score_history_threshold=10000,
                                                 log_dir=log_dir6).run(100000000))
            elif self.opponent == "MOISMCTSWithRandomRolloutsAgent":
                log_dir7 = self.log_dir_root + "TabularQLearningAgent_VS_MOISMCTSWithRandomRolloutsAgent_" + self.time
                print(log_dir7)
                print(TensorboardTicTacToeRunner(TabularQLearningAgent(),
                                                 MOISMCTSWithRandomRolloutsAgent(100,
                                                                                 SafeTicTacToeRunner(RandomAgent(),
                                                                                                     RandomAgent())),
                                                 log_and_reset_score_history_threshold=10000,
                                                 log_dir=log_dir7).run(1000000000))
            elif self.opponent == "MOISMCTSWithRandomRolloutsExpertThenApprenticeAgent":
                log_dir8 = self.log_dir_root + "TabularQLearningAgent_VS_MOISMCTSWithRandomRolloutsExpertThenApprenticeAgent_" + self.time
                print(log_dir8)
                print(TensorboardTicTacToeRunner(TabularQLearningAgent(),
                                                 MOISMCTSWithRandomRolloutsExpertThenApprenticeAgent(100,
                                                                                                     SafeTicTacToeRunner(
                                                                                                         RandomAgent(),
                                                                                                         RandomAgent()),9,9),
                                                 log_and_reset_score_history_threshold=10000,
                                                 log_dir=log_dir8).run(1000000000))
            elif self.opponent == "MOISMCTSWithValueNetworkAgent":
                log_dir9 = self.log_dir_root + "TabularQLearningAgent_VS_MOISMCTSWithValueNetworkAgent_" + self.time
                print(log_dir9)
                print(TensorboardTicTacToeRunner(TabularQLearningAgent(),
                                                 MOISMCTSWithValueNetworkAgent(100,
                                                                               SafeTicTacToeRunner(RandomAgent(),
                                                                                                   RandomAgent())),
                                                 log_and_reset_score_history_threshold=10000,
                                                 log_dir=log_dir9).run(1000000000))
            elif self.opponent == "DoubleQLearningAgent":
                log_dir10 = self.log_dir_root + "TabularQLearningAgent_VS_DoubleQLearningAgent_" + self.time
                print(log_dir10)
                print(TensorboardTicTacToeRunner(TabularQLearningAgent(),
                                                 DoubleQLearningAgent(9,9),
                                                 log_and_reset_score_history_threshold=10000,
                                                 log_dir=log_dir10).run(1000000000))
            elif self.opponent == "RandomRolloutAgent":
                nb_rollouts = 3
                log_dir11 = self.log_dir_root + "TabularQLearningAgent_VS_RandomRolloutAgent(" + str(nb_rollouts) + ")_" + self.time
                print(log_dir11)
                print(TensorboardTicTacToeRunner(TabularQLearningAgent(),
                                                 RandomRolloutAgent(3,
                                                     SafeTicTacToeRunner(
                                                         RandomAgent(),
                                                         RandomAgent())),
                                                 log_and_reset_score_history_threshold=10000,
                                                 log_dir=log_dir11).run(1000000000))
            else:
                print("Unknown opponent")


    log_dir_root = "./logs/TabularQLearningAgent_vs_all/"
    # Création des threads
    # thread_1 = Bench(log_dir_root=log_dir_root, opponent="RandomAgent")
    thread_2 = Bench(log_dir_root=log_dir_root, opponent="TabularQLearningAgent")
    thread_3 = Bench(log_dir_root=log_dir_root, opponent="DeepQLearningAgent")
    thread_4 = Bench(log_dir_root=log_dir_root, opponent="ReinforceClassicAgent")
    thread_5 = Bench(log_dir_root=log_dir_root, opponent="ReinforceClassicWithMultipleTrajectoriesAgent")
    thread_6 = Bench(log_dir_root=log_dir_root, opponent="PPOWithMultipleTrajectoriesMultiOutputsAgent")
    thread_7 = Bench(log_dir_root=log_dir_root, opponent="MOISMCTSWithRandomRolloutsAgent")
    # thread_8 = Bench(log_dir_root=log_dir_root, opponent="MOISMCTSWithRandomRolloutsExpertThenApprenticeAgent")
    # thread_9 = Bench(log_dir_root=log_dir_root, opponent="MOISMCTSWithValueNetworkAgent")
    # thread_10 = Bench(log_dir_root=log_dir_root, opponent="DoubleQLearningAgent")
    thread_11 = Bench(log_dir_root=log_dir_root, opponent="RandomRolloutAgent")

    # thread_1.start()
    thread_2.start()
    thread_3.start()
    thread_4.start()
    thread_5.start()
    thread_6.start()
    thread_7.start()
    # thread_8.start()
    # thread_9.start()
    # thread_10.start()
    thread_11.start()

    # thread_1.join()
    thread_2.join()
    thread_3.join()
    thread_4.join()
    thread_5.join()
    thread_6.join()
    thread_7.join()
    # thread_8.join()
    # thread_9.join()
    # thread_10.join()
    thread_11.join()

    print("All runs are finished !")