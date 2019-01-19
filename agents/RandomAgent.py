import random

from environments import InformationState
from environments.Agent import Agent


class RandomAgent(Agent):
    def observe(self, reward: float, terminal: bool) -> None:
        pass

    def act(self, player_index: int, information_state: InformationState, available_actions: 'Iterable[int]') -> int:
        action_count = len(available_actions)
        if action_count == 1:
            return available_actions[0]
        else:
            return available_actions[random.randint(0, action_count - 1)]

    def toString(self) -> str:
        return "RandomAgent"

