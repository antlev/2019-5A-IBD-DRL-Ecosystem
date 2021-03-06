import os
from environments import Agent
from environments.GameRunner import GameRunner
from games.windjammers.WindJammersGameState import WindJammersGameState

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class SafeWindJammersRunner(GameRunner):
    def __init__(self, agent1: Agent, agent2: Agent):
        self.agents = (agent1, agent2)

    def run(self, max_rounds: int = -1,
            initial_game_state: WindJammersGameState = WindJammersGameState()):
        episode_id = 1

        while episode_id < max_rounds or max_rounds == -1:
            gs = initial_game_state.copy_game_state()
            terminal = False
            round_step = 0
            while not terminal:
                current_player = gs.get_current_player_id()
                action = 0
                if current_player != -1:
                    action_ids = gs.get_available_actions_id_for_player(current_player)
                    info_state = gs.get_information_state_for_player(current_player)
                    action = self.agents[current_player].act(current_player,
                                                             info_state,
                                                             action_ids)

                # WARNING : Two Players Zero Sum Game Hypothesis
                (gs, score, terminal) = gs.step(current_player, action)
                self.agents[0].observe(score, terminal)
                self.agents[1].observe(-score, terminal)
            round_step += 1
            episode_id += 1
