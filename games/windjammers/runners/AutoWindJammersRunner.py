import sys

from agents.MOISMCTSWithRandomRolloutsExpertThenApprenticeAgent import \
    MOISMCTSWithRandomRolloutsExpertThenApprenticeAgent
from agents.MOISMCTSWithValueNetworkAgent import MOISMCTSWithValueNetworkAgent

sys.path.append('/home/tredzone/antoine/2019-5A-IBD-DRL-Ecosystem/')
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

class Bench1(Thread):
    def __init__(self):
        Thread.__init__(self)

    def run(self):
        log_dir = "./logs/bastille/PPOWithMultipleTrajectoriesMultiOutputs_Vs_TabularQLearning/" + str(time())
        print(str(log_dir))
        print(TensorboardWindJammersRunner(
            PPOWithMultipleTrajectoriesMultiOutputsAgent.PPOWithMultipleTrajectoriesMultiOutputsAgent(8, 12),
            TabularQLearningAgent(),
            checkpoint=100,
            log_dir=log_dir).run(1000000))


class Bench2(Thread):
    def __init__(self):
        Thread.__init__(self)

    def run(self):
        log_dir = "./logs/bastille/ReinforceWithMultipleTraj_Vs_TabularQLearning/" + str(time())
        print(str(log_dir))
        print(TensorboardWindJammersRunner(
            ReinforceClassicWithMultipleTrajectoriesAgent(8, 12),
            TabularQLearningAgent(),
            checkpoint=100,
            log_dir=log_dir).run(1000000))


class Bench3(Thread):
    def __init__(self):
        Thread.__init__(self)

    def run(self):
        log_dir = "./logs/bastille/DeepQLearning_Vs_TabularQLearning/" + str(time())
        print(str(log_dir))
        print(TensorboardWindJammersRunner(
            DeepQLearningAgent(8, 12),
            TabularQLearningAgent(),
            checkpoint=100,
            log_dir=log_dir).run(1000000))


class Bench4(Thread):
    def __init__(self):
        Thread.__init__(self)

    def run(self):
        log_dir = "./logs/bastille/MOISMCTSWithRandomRollouts_Vs_TabularQLearning/" + str(time())
        print(str(log_dir))
        print(TensorboardWindJammersRunner(
            MOISMCTSWithRandomRolloutsAgent(100, SafeWindJammersRunner(RandomAgent(), RandomAgent())),
            TabularQLearningAgent(),
            checkpoint=100,
            log_dir=log_dir).run(1000000))


class Bench5(Thread):
    def __init__(self):
        Thread.__init__(self)

    def run(self):
        log_dir = "./logs/bastille/ReinforceClassicWithMultipleTrajectories_Vs_TabularQLearningAgent/" + str(time())
        print(str(log_dir))
        print(TensorboardWindJammersRunner(
            ReinforceClassicWithMultipleTrajectoriesAgent(8, 12),
            TabularQLearningAgent(),
            checkpoint=100,
            log_dir=log_dir).run(1000000))


class Bench6(Thread):
    def __init__(self):
        Thread.__init__(self)

    def run(self):
        log_dir = "./logs/bastille/Random_Vs_TabularQLearningAgent/" + str(time())
        print(str(log_dir))
        print(TensorboardWindJammersRunner(
            RandomAgent(),
            TabularQLearningAgent(),
            checkpoint=100,
            log_dir=log_dir).run(1000000))


class Bench7(Thread):
    def __init__(self):
        Thread.__init__(self)

    def run(self):
        log_dir = "./logs/bastille/Random_Vs_RandomRollout_100/" + str(time())
        print(str(log_dir))
        print(TensorboardWindJammersRunner(
            RandomAgent(),
            RandomRolloutAgent(100, SafeWindJammersRunner(RandomAgent(), RandomAgent())),
            checkpoint=100,
            log_dir=log_dir).run(1000000))


class Bench8(Thread):
    def __init__(self):
        Thread.__init__(self)

    def run(self):
        log_dir = "./logs/bastille/Random_Vs_DeepQLearning/" + str(time())
        print(str(log_dir))
        print(TensorboardWindJammersRunner(
            RandomAgent(),
            DeepQLearningAgent(8, 12),
            checkpoint=100,
            log_dir=log_dir).run(1000000))


class Bench9(Thread):
    def __init__(self):
        Thread.__init__(self)

    def run(self):
        log_dir = "./logs/bastille/Random_Vs_DoubleQLearning/" + str(time())
        print(str(log_dir))
        print(TensorboardWindJammersRunner(
            RandomAgent(),
            DoubleQLearningAgent(),
            checkpoint=100,
            log_dir=log_dir).run(1000000))


class Bench10(Thread):
    def __init__(self):
        Thread.__init__(self)

    def run(self):
        log_dir = "./logs/bastille/Random_Vs_ReinforceClassic/" + str(time())
        print(str(log_dir))
        print(TensorboardWindJammersRunner(
            RandomAgent(),
            ReinforceClassicAgent(8, 12),
            checkpoint=100,
            log_dir=log_dir).run(1000000))


class Bench11(Thread):
    def __init__(self):
        Thread.__init__(self)

    def run(self):
        log_dir = "./logs/bastille/Random_Vs_ReinforceClassicWithMultipleTrajectories/" + str(time())
        print(str(log_dir))
        print(TensorboardWindJammersRunner(
            RandomAgent(),
            ReinforceClassicWithMultipleTrajectoriesAgent(8, 12),
            checkpoint=100,
            log_dir=log_dir).run(1000000))


class Bench12(Thread):
    def __init__(self):
        Thread.__init__(self)

    def run(self):
        log_dir = "./logs/bastille/Random_Vs_PPOWithMultipleTrajectoriesMultiOutputs" + str(time())
        print(str(log_dir))
        print(TensorboardWindJammersRunner(
            RandomAgent(),
            PPOWithMultipleTrajectoriesMultiOutputsAgent(8, 12),
            checkpoint=100,
            log_dir=log_dir).run(1000000))


class Bench13(Thread):
    def __init__(self):
        Thread.__init__(self)

    def run(self):
        log_dir = "./logs/bastille/Random_Vs_MOISMCTSWithRandomRollouts/" + str(time())
        print(str(log_dir))
        print(TensorboardWindJammersRunner(
            RandomAgent(),
            MOISMCTSWithRandomRolloutsAgent(100, SafeWindJammersRunner(RandomAgent(), RandomAgent())),
            checkpoint=100,
            log_dir=log_dir).run(1000000))


class Bench14(Thread):
    def __init__(self):
        Thread.__init__(self)

    def run(self):
        log_dir = "./logs/bastille/Random_Vs_MOISMCTSWithRandomRolloutsExpertThenApprentice/" + str(time())
        print(str(log_dir))
        print(TensorboardWindJammersRunner(
            RandomAgent(),
            MOISMCTSWithRandomRolloutsExpertThenApprenticeAgent(100,
                                                                SafeWindJammersRunner(RandomAgent(), RandomAgent()), 8,
                                                                12),
            checkpoint=100,
            log_dir=log_dir).run(1000000))


class Bench15(Thread):
    def __init__(self):
        Thread.__init__(self)

    def run(self):
        log_dir = "./logs/bastille/Random_Vs_MOISMCTSWithValueNetwork/" + str(time())
        print(str(log_dir))
        print(TensorboardWindJammersRunner(
            RandomAgent(),
            MOISMCTSWithValueNetworkAgent(100, SafeWindJammersRunner(RandomAgent(), RandomAgent())),
            checkpoint=100,
            log_dir=log_dir).run(1000000))


class Bench16(Thread):
    def __init__(self):
        Thread.__init__(self)

    def run(self):
        log_dir = "./logs/bastille/TabularQLearning_RandomRollout_100/" + str(time())
        print(str(log_dir))
        print(TensorboardWindJammersRunner(
            TabularQLearningAgent(),
            RandomRolloutAgent(100, SafeWindJammersRunner(RandomAgent(), RandomAgent())),
            checkpoint=100,
            log_dir=log_dir).run(1000000))


class Bench17(Thread):
    def __init__(self):
        Thread.__init__(self)

    def run(self):
        log_dir = "./logs/bastille/TabularQLearning_DeepQLearning/" + str(time())
        print(str(log_dir))
        print(TensorboardWindJammersRunner(
            TabularQLearningAgent(),
            DeepQLearningAgent(8, 12),
            checkpoint=100,
            log_dir=log_dir).run(1000000))


class Bench18(Thread):
    def __init__(self):
        Thread.__init__(self)

    def run(self):
        log_dir = "./logs/bastille/TabularQLearning_DoubleQLearning/" + str(time())
        print(str(log_dir))
        print(TensorboardWindJammersRunner(
            TabularQLearningAgent(),
            DoubleQLearningAgent(),
            checkpoint=100,
            log_dir=log_dir).run(1000000))


class Bench19(Thread):
    def __init__(self):
        Thread.__init__(self)

    def run(self):
        log_dir = "./logs/bastille/TabularQLearning_ReinforceClassic/" + str(time())
        print(str(log_dir))
        print(TensorboardWindJammersRunner(
            TabularQLearningAgent(),
            ReinforceClassicAgent(8, 12),
            checkpoint=100,
            log_dir=log_dir).run(1000000))


class Bench20(Thread):
    def __init__(self):
        Thread.__init__(self)

    def run(self):
        log_dir = "./logs/bastille/TabularQLearning_ReinforceClassicWithMultipleTrajectories/" + str(time())
        print(str(log_dir))
        print(TensorboardWindJammersRunner(
            TabularQLearningAgent(),
            ReinforceClassicWithMultipleTrajectoriesAgent(8, 12),
            checkpoint=100,
            log_dir=log_dir).run(1000000))


class Bench21(Thread):
    def __init__(self):
        Thread.__init__(self)

    def run(self):
        log_dir = "./logs/bastille/TabularQLearning_PPOWithMultipleTrajectoriesMultiOutputs" + str(time())
        print(str(log_dir))
        print(TensorboardWindJammersRunner(
            TabularQLearningAgent(),
            PPOWithMultipleTrajectoriesMultiOutputsAgent(8, 12),
            checkpoint=100,
            log_dir=log_dir).run(1000000))


class Bench22(Thread):
    def __init__(self):
        Thread.__init__(self)

    def run(self):
        log_dir = "./logs/bastille/TabularQLearning_MOISMCTSWithRandomRollouts/" + str(time())
        print(str(log_dir))
        print(TensorboardWindJammersRunner(
            TabularQLearningAgent(),
            MOISMCTSWithRandomRolloutsAgent(100, SafeWindJammersRunner(RandomAgent(), RandomAgent())),
            checkpoint=100,
            log_dir=log_dir).run(1000000))


class Bench23(Thread):
    def __init__(self):
        Thread.__init__(self)

    def run(self):
        log_dir = "./logs/bastille/TabularQLearning_MOISMCTSWithRandomRolloutsExpertThenApprentice/" + str(time())
        print(str(log_dir))
        print(TensorboardWindJammersRunner(
            TabularQLearningAgent(),
            MOISMCTSWithRandomRolloutsExpertThenApprenticeAgent(100,
                                                                SafeWindJammersRunner(RandomAgent(), RandomAgent()), 8,
                                                                12),
            checkpoint=100,
            log_dir=log_dir).run(1000000))


class Bench24(Thread):
    def __init__(self):
        Thread.__init__(self)

    def run(self):
        log_dir = "./logs/bastille/TabularQLearning_MOISMCTSWithValueNetwork/" + str(time())
        print(str(log_dir))
        print(TensorboardWindJammersRunner(
            TabularQLearningAgent(),
            MOISMCTSWithValueNetworkAgent(100, SafeWindJammersRunner(RandomAgent(), RandomAgent())),
            checkpoint=100,
            log_dir=log_dir).run(1000000))


if __name__ == "__main__":
    size = 10000000  # Number of random numbers to add
    procs = 24  # Number of processes to create

    # Create a list of jobs and then iterate through
    # the number of processes appending each process to
    # the job list
    jobs = []
    for i in range(0, procs):
        out_list = list()
        process = Process(target=list_append,
                                          args=(size, i, out_list))
        jobs.append(process)

    # Start the processes (i.e. calculate the random number lists)
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

