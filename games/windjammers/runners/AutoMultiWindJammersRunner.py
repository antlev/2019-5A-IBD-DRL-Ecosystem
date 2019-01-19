import sys
sys.path.append('/home/tredzone/antoine/2019-5A-IBD-DRL-Ecosystem/')

from agents.MOISMCTSWithRandomRolloutsExpertThenApprenticeAgent import \
    MOISMCTSWithRandomRolloutsExpertThenApprenticeAgent
from agents.MOISMCTSWithValueNetworkAgent import MOISMCTSWithValueNetworkAgent

from threading import Thread
from time import time
from agents.MOISMCTSWithRandomRolloutsAgent import MOISMCTSWithRandomRolloutsAgent
from agents.DeepQLearningAgent import DeepQLearningAgent
from agents.DoubleQLearningAgent import DoubleQLearningAgent
from agents.PPOWithMultipleTrajectoriesMultiOutputsAgent import PPOWithMultipleTrajectoriesMultiOutputsAgent
from agents.RandomRolloutAgent import RandomRolloutAgent
from agents.ReinforceClassicAgent import ReinforceClassicAgent
from agents.ReinforceClassicWithMultipleTrajectoriesAgent import ReinforceClassicWithMultipleTrajectoriesAgent
from agents.TabularQLearningAgent import TabularQLearningAgent
from agents.RandomAgent import RandomAgent
from games.windjammers.runners.TensorboardWindJammersRunner import TensorboardWindJammersRunner
from games.windjammers.runners.SafeWindJammersRunner import SafeWindJammersRunner

from multiprocessing import Process, current_process

def run():


    log_dir = "./logs/bastilleMP/ReinforceWithMultipleTraj_Vs_TabularQLearning/" + str(time())
    print(str(log_dir))
    print(TensorboardWindJammersRunner(
        ReinforceClassicWithMultipleTrajectoriesAgent(8, 12),
        TabularQLearningAgent(),
        checkpoint=100,
        log_dir=log_dir).run(100000))


    log_dir = "./logs/bastilleMP/DeepQLearning_Vs_TabularQLearning/" + str(time())
    print(str(log_dir))
    print(TensorboardWindJammersRunner(
        DeepQLearningAgent(8, 12),
        TabularQLearningAgent(),
        checkpoint=100,
        log_dir=log_dir).run(100000))


    log_dir = "./logs/bastilleMP/MOISMCTSWithRandomRollouts_Vs_TabularQLearning/" + str(time())
    print(str(log_dir))
    print(TensorboardWindJammersRunner(
        MOISMCTSWithRandomRolloutsAgent(100, SafeWindJammersRunner(RandomAgent(), RandomAgent())),
        TabularQLearningAgent(),
        checkpoint=100,
        log_dir=log_dir).run(100000))


    log_dir = "./logs/bastilleMP/ReinforceClassicWithMultipleTrajectories_Vs_TabularQLearningAgent/" + str(time())
    print(str(log_dir))
    print(TensorboardWindJammersRunner(
        ReinforceClassicWithMultipleTrajectoriesAgent(8, 12),
        TabularQLearningAgent(),
        checkpoint=100,
        log_dir=log_dir).run(100000))


    log_dir = "./logs/bastilleMP/Random_Vs_TabularQLearningAgent/" + str(time())
    print(str(log_dir))
    print(TensorboardWindJammersRunner(
        RandomAgent(),
        TabularQLearningAgent(),
        checkpoint=100,
        log_dir=log_dir).run(100000))


    log_dir = "./logs/bastilleMP/Random_Vs_RandomRollout_100/" + str(time())
    print(str(log_dir))
    print(TensorboardWindJammersRunner(
        RandomAgent(),
        RandomRolloutAgent(100, SafeWindJammersRunner(RandomAgent(), RandomAgent())),
        checkpoint=100,
        log_dir=log_dir).run(100000))


    log_dir = "./logs/bastilleMP/Random_Vs_DeepQLearning/" + str(time())
    print(str(log_dir))
    print(TensorboardWindJammersRunner(
        RandomAgent(),
        DeepQLearningAgent(8, 12),
        checkpoint=100,
        log_dir=log_dir).run(100000))


    log_dir = "./logs/bastilleMP/Random_Vs_DoubleQLearning/" + str(time())
    print(str(log_dir))
    print(TensorboardWindJammersRunner(
        RandomAgent(),
        DoubleQLearningAgent(),
        checkpoint=100,
        log_dir=log_dir).run(100000))



    log_dir = "./logs/bastilleMP/Random_Vs_ReinforceClassic/" + str(time())
    print(str(log_dir))
    print(TensorboardWindJammersRunner(
        RandomAgent(),
        ReinforceClassicAgent(8, 12),
        checkpoint=100,
        log_dir=log_dir).run(100000))



    log_dir = "./logs/bastilleMP/Random_Vs_ReinforceClassicWithMultipleTrajectories/" + str(time())
    print(str(log_dir))
    print(TensorboardWindJammersRunner(
        RandomAgent(),
        ReinforceClassicWithMultipleTrajectoriesAgent(8, 12),
        checkpoint=100,
        log_dir=log_dir).run(100000))



    log_dir = "./logs/bastilleMP/Random_Vs_PPOWithMultipleTrajectoriesMultiOutputs" + str(time())
    print(str(log_dir))
    print(TensorboardWindJammersRunner(
        RandomAgent(),
        PPOWithMultipleTrajectoriesMultiOutputsAgent(8, 12),
        checkpoint=100,
        log_dir=log_dir).run(100000))



    log_dir = "./logs/bastilleMP/Random_Vs_MOISMCTSWithRandomRollouts/" + str(time())
    print(str(log_dir))
    print(TensorboardWindJammersRunner(
        RandomAgent(),
        MOISMCTSWithRandomRolloutsAgent(100, SafeWindJammersRunner(RandomAgent(), RandomAgent())),
        checkpoint=100,
        log_dir=log_dir).run(100000))



    log_dir = "./logs/bastilleMP/Random_Vs_MOISMCTSWithRandomRolloutsExpertThenApprentice/" + str(time())
    print(str(log_dir))
    print(TensorboardWindJammersRunner(
        RandomAgent(),
        MOISMCTSWithRandomRolloutsExpertThenApprenticeAgent(100,
                                                            SafeWindJammersRunner(RandomAgent(), RandomAgent()), 8,
                                                            12),
        checkpoint=100,
        log_dir=log_dir).run(100000))



    log_dir = "./logs/bastilleMP/Random_Vs_MOISMCTSWithValueNetwork/" + str(time())
    print(str(log_dir))
    print(TensorboardWindJammersRunner(
        RandomAgent(),
        MOISMCTSWithValueNetworkAgent(100, SafeWindJammersRunner(RandomAgent(), RandomAgent())),
        checkpoint=100,
        log_dir=log_dir).run(100000))



    log_dir = "./logs/bastilleMP/TabularQLearning_RandomRollout_100/" + str(time())
    print(str(log_dir))
    print(TensorboardWindJammersRunner(
        TabularQLearningAgent(),
        RandomRolloutAgent(100, SafeWindJammersRunner(RandomAgent(), RandomAgent())),
        checkpoint=100,
        log_dir=log_dir).run(100000))



    log_dir = "./logs/bastilleMP/TabularQLearning_DeepQLearning/" + str(time())
    print(str(log_dir))
    print(TensorboardWindJammersRunner(
        TabularQLearningAgent(),
        DeepQLearningAgent(8, 12),
        checkpoint=100,
        log_dir=log_dir).run(100000))



    log_dir = "./logs/bastilleMP/TabularQLearning_DoubleQLearning/" + str(time())
    print(str(log_dir))
    print(TensorboardWindJammersRunner(
        TabularQLearningAgent(),
        DoubleQLearningAgent(),
        checkpoint=100,
        log_dir=log_dir).run(100000))



    log_dir = "./logs/bastilleMP/TabularQLearning_ReinforceClassic/" + str(time())
    print(str(log_dir))
    print(TensorboardWindJammersRunner(
        TabularQLearningAgent(),
        ReinforceClassicAgent(8, 12),
        checkpoint=100,
        log_dir=log_dir).run(100000))



    log_dir = "./logs/bastilleMP/TabularQLearning_ReinforceClassicWithMultipleTrajectories/" + str(time())
    print(str(log_dir))
    print(TensorboardWindJammersRunner(
        TabularQLearningAgent(),
        ReinforceClassicWithMultipleTrajectoriesAgent(8, 12),
        checkpoint=100,
        log_dir=log_dir).run(100000))



    log_dir = "./logs/bastilleMP/TabularQLearning_PPOWithMultipleTrajectoriesMultiOutputs" + str(time())
    print(str(log_dir))
    print(TensorboardWindJammersRunner(
        TabularQLearningAgent(),
        PPOWithMultipleTrajectoriesMultiOutputsAgent(8, 12),
        checkpoint=100,
        log_dir=log_dir).run(100000))



    log_dir = "./logs/bastilleMP/TabularQLearning_MOISMCTSWithRandomRollouts/" + str(time())
    print(str(log_dir))
    print(TensorboardWindJammersRunner(
        TabularQLearningAgent(),
        MOISMCTSWithRandomRolloutsAgent(100, SafeWindJammersRunner(RandomAgent(), RandomAgent())),
        checkpoint=100,
        log_dir=log_dir).run(100000))



    log_dir = "./logs/bastilleMP/TabularQLearning_MOISMCTSWithRandomRolloutsExpertThenApprentice/" + str(time())
    print(str(log_dir))
    print(TensorboardWindJammersRunner(
        TabularQLearningAgent(),
        MOISMCTSWithRandomRolloutsExpertThenApprenticeAgent(100,
                                                            SafeWindJammersRunner(RandomAgent(), RandomAgent()), 8,
                                                            12),
        checkpoint=100,
        log_dir=log_dir).run(100000))



    log_dir = "./logs/bastilleMP/TabularQLearning_MOISMCTSWithValueNetwork/" + str(time())
    print(str(log_dir))
    print(TensorboardWindJammersRunner(
        TabularQLearningAgent(),
        MOISMCTSWithValueNetworkAgent(100, SafeWindJammersRunner(RandomAgent(), RandomAgent())),
        checkpoint=100,
        log_dir=log_dir).run(100000))


if __name__ == "__main__":
    procs = 24

    jobs = []
    for i in range(0, procs):
        out_list = list()
        process = Process(target=run)
        jobs.append(process)

    for j in jobs:
        j.start()

    # Ensure all of the processes have finished
    for j in jobs:
        j.join()

# AGENTS EXAMPLES :
# CommandLineAgent()
# RandomAgent()
# RandomRolloutAgent(3, SafeWindJammersRunner(RandomAgent(), RandomAgent()))
# TabularQLearningAgent()
# DeepQLearningAgent(8,12)
# ReinforceClassicAgent(8,12)
# ReinforceClassicWithMultipleTrajectoriesAgent(8,12)
# PPOWithMultipleTrajectoriesMultiOutputsAgent(8,12)
# MOISMCTSWithRandomRolloutsAgent(100, SafeWindJammersRunner(RandomAgent(), RandomAgent()))
# MOISMCTSWithRandomRolloutsExpertThenApprenticeAgent(100, SafeWindJammersRunner(RandomAgent(), RandomAgent()),8,12)
# MOISMCTSWithValueNetworkAgent(100, SafeWindJammersRunner(RandomAgent(), RandomAgent()))

