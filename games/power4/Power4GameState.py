from copy import deepcopy
import numpy as np

from environments import InformationState
from environments.GameState import GameState
from games.power4.Power4InformationState import Power4InformationState


class Power4GameState(GameState):

    def __init__(self):
        self.current_player = 0
        self.board = np.array(
            (
                (0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0)
             )
        )

    def step(self, player_id: int, action_id: int) -> \
            ('GameState', float, bool):
        if self.current_player != player_id :
            raise Exception("This is not this player turn !")
        val = self.board[action_id // 6][action_id % 7]
        if (val != 0):
            raise Exception("Player can't play at specified position !")

        self.board[action_id // 6][action_id % 7] = \
            1 if player_id == 0 else -1

        (score, terminal) = self.compute_current_score_and_end_game_more_efficient()

        self.current_player = (self.current_player + 1) % 2
        return (self, score, terminal)

    def compute_current_score_and_end_game(self):
        board = self.board
        for i in range(6):
            for j in range(7):
                if j+3 <= 6:
                    if board[i][j] + board[i][j+1] + board[i][j+2] + board[i][j+3] == 4:
                        return 1, True
                    if board[i][j] + board[i][j+1] + board[i][j+2] + board[i][j+3] == -4:
                        return -1, True
                if i+3 <= 5:
                    if board[i][j] + board[i+1][j] + board[i+2][j] + board[i+3][j] == 4:
                        return 1, True
                    if board[i][j] + board[i+1][j] + board[i+2][j] + board[i+3][j] == -4:
                        return -1, True
        # ???????
        if 0 in board:
            return 0.0, False
        return 0.0, True

    def get_player_count(self) -> int:
        return 2

    def get_current_player_id(self) -> int:
        return self.current_player

    def get_information_state_for_player(self, player_id: int) -> 'InformationState':
        return Power4InformationState(self.current_player,
                                         self.board.copy())

    def get_available_actions_id_for_player(self, player_id: int) -> 'Iterable(int)':
        if player_id != self.current_player:
            return []
        return list(filter(lambda i: self.board[i // 6][i % 7] == 0, range(0, 42)))

    def __str__(self):
        str = ""
        for i in range(5):
            for j in range(6):
                val = self.board[i][j]
                str += "_" if val == 0 else (
                    "0" if val == 1 else
                    "X"
                )
            str += "\n"
        return str

    def copy_game_state(self):
        gs = Power4GameState()
        gs.board = self.board.copy()
        gs.current_player = self.current_player
        return gs


if __name__ == "__main__":
    gs = Power4GameState()
    print(gs.get_available_actions_id_for_player(gs.get_current_player_id()))
    print(gs)
    gs.step(0, 0)
    print(gs)
    gs.step(1, 2)
    print(gs)
    gs.step(0, 2)
