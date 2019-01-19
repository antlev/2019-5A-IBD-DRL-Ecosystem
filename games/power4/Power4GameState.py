from copy import deepcopy
import numpy as np

from environments import InformationState
from environments.GameState import GameState
from games.power4.Power4InformationState import Power4InformationState


class Power4GameState(GameState):

    def __init__(self):
        self.current_player = 0
        self.nb_lines = 7
        self.nb_cols = 7
        self.board = np.array(
            (
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0
             )
        )

    def help(self):
        print("---------- REMINDER ----------")
        gb = ""
        for i in range(7):
            for j in range(7):
                if j + 7 * (7-(i+1)) < 10:
                    gb += "  " + str(j + 7 * (7-(i+1))) + " "
                else:
                    gb += " " + str(j + 7 * (7-(i+1))) + " "
            gb += "\n"
        print(gb)


    def step(self, player_id: int, action_id: int) -> \
            ('GameState', float, bool):
        if self.current_player != player_id :
            raise Exception("This is not this player turn !")
        val = self.board[action_id]
        if (val != 0):
            raise Exception("Player can't play at specified position !")
        self.board[action_id] = \
            1 if player_id == 0 else -1
        (score, terminal) = self.compute_current_score_and_end_game()
        self.current_player = (self.current_player + 1) % 2
        return (self, score, terminal)

    def compute_current_score_and_end_game(self):
        successive_position = [0,0]
        # LINES
        for i in range(self.nb_lines):
            for j in range(self.nb_cols):
                if self.board[j*self.nb_cols+i] == 1:
                    successive_position[1] = 0
                    successive_position[0] += 1
                    if successive_position[0] == 4:
                        return 1.0, True
                elif self.board[j * self.nb_cols + i] == 1:
                    successive_position[0] = 0
                    successive_position[1] += 1
                    if successive_position[0] == 4:
                        return -1.0, True
                else:
                    successive_position[0] = 0
                    successive_position[1] = 0
        # COLUMNS
        for j in range(self.nb_lines):
            for i in range(self.nb_cols):
                if self.board[j*self.nb_cols+i] == 1:
                    successive_position[1] = 0
                    successive_position[0] += 1
                    if successive_position[0] == 4:
                        return 1.0, True
                elif self.board[j * self.nb_cols + i] == 1:
                    successive_position[0] = 0
                    successive_position[1] += 1
                    if successive_position[0] == 4:
                        return -1.0, True
                else:
                    successive_position[0] = 0
                    successive_position[1] = 0
        # DIAGS LEFT -> RIGHT
        case1 = 21
        case2 = 29
        case3 = 37
        case4 = 45
        if self.board[case1] == self.board[case2] and self.board[case1] == self.board[case3]  and self.board[case1] == self.board[case4] and self.board[case1] != 0:
            return self.board[case1], True
        case1 -= self.nb_cols
        case2 -= self.nb_cols
        case3 -= self.nb_cols
        case4 -= self.nb_cols
        if self.board[case1] == self.board[case2] and self.board[case1] == self.board[case3]  and self.board[case1] == self.board[case4] and self.board[case1] != 0:
            return self.board[case1], True
        case1 += self.nb_cols + 1
        case2 += self.nb_cols + 1
        case3 += self.nb_cols + 1
        case4 += self.nb_cols + 1
        if self.board[case1] == self.board[case2] and self.board[case1] == self.board[case3]  and self.board[case1] == self.board[case4] and self.board[case1] != 0:
            return self.board[case1], True
        case1 -= 2*self.nb_cols + 1
        case2 -= 2*self.nb_cols + 1
        case3 -= 2*self.nb_cols + 1
        case4 -= 2*self.nb_cols + 1
        if self.board[case1] == self.board[case2] and self.board[case1] == self.board[case3]  and self.board[case1] == self.board[case4] and self.board[case1] != 0:
            return self.board[case1], True
        case1 += self.nb_cols + 1
        case2 += self.nb_cols + 1
        case3 += self.nb_cols + 1
        case4 += self.nb_cols + 1
        if self.board[case1] == self.board[case2] and self.board[case1] == self.board[case3]  and self.board[case1] == self.board[case4] and self.board[case1] != 0:
            return self.board[case1], True
        case1 += self.nb_cols + 1
        case2 += self.nb_cols + 1
        case3 += self.nb_cols + 1
        case4 += self.nb_cols + 1
        if self.board[case1] == self.board[case2] and self.board[case1] == self.board[case3]  and self.board[case1] == self.board[case4] and self.board[case1] != 0:
            return self.board[case1], True
        case1 -= 3*self.nb_cols + 2
        case2 -= 3*self.nb_cols + 2
        case3 -= 3*self.nb_cols + 2
        case4 -= 3*self.nb_cols + 2
        if self.board[case1] == self.board[case2] and self.board[case1] == self.board[case3] and self.board[case1] == \
                self.board[case4] and self.board[case1] != 0:
            return self.board[case1], True
        case1 += self.nb_cols + 1
        case2 += self.nb_cols + 1
        case3 += self.nb_cols + 1
        case4 += self.nb_cols + 1
        if self.board[case1] == self.board[case2] and self.board[case1] == self.board[case3] and self.board[case1] == \
                self.board[case4] and self.board[case1] != 0:
            return self.board[case1], True
        case1 += self.nb_cols + 1
        case2 += self.nb_cols + 1
        case3 += self.nb_cols + 1
        case4 += self.nb_cols + 1
        if self.board[case1] == self.board[case2] and self.board[case1] == self.board[case3] and self.board[case1] == \
                self.board[case4] and self.board[case1] != 0:
            return self.board[case1], True
        case1 += self.nb_cols + 1
        case2 += self.nb_cols + 1
        case3 += self.nb_cols + 1
        case4 += self.nb_cols + 1
        if self.board[case1] == self.board[case2] and self.board[case1] == self.board[case3] and self.board[case1] == \
                self.board[case4] and self.board[case1] != 0:
            return self.board[case1], True
        case1 -= 3*self.nb_cols + 2
        case2 -= 3*self.nb_cols + 2
        case3 -= 3*self.nb_cols + 2
        case4 -= 3*self.nb_cols + 2
        if self.board[case1] == self.board[case2] and self.board[case1] == self.board[case3] and self.board[case1] == \
                self.board[case4] and self.board[case1] != 0:
            return self.board[case1], True
        case1 += self.nb_cols + 1
        case2 += self.nb_cols + 1
        case3 += self.nb_cols + 1
        case4 += self.nb_cols + 1
        if self.board[case1] == self.board[case2] and self.board[case1] == self.board[case3] and self.board[case1] == \
                self.board[case4] and self.board[case1] != 0:
            return self.board[case1], True
        case1 += self.nb_cols + 1
        case2 += self.nb_cols + 1
        case3 += self.nb_cols + 1
        case4 += self.nb_cols + 1
        if self.board[case1] == self.board[case2] and self.board[case1] == self.board[case3] and self.board[case1] == \
                self.board[case4] and self.board[case1] != 0:
            return self.board[case1], True
        case1 -= 2 * self.nb_cols + 1
        case2 -= 2 * self.nb_cols + 1
        case3 -= 2 * self.nb_cols + 1
        case4 -= 2 * self.nb_cols + 1
        if self.board[case1] == self.board[case2] and self.board[case1] == self.board[case3] and self.board[case1] == \
                self.board[case4] and self.board[case1] != 0:
            return self.board[case1], True
        case1 += self.nb_cols + 1
        case2 += self.nb_cols + 1
        case3 += self.nb_cols + 1
        case4 += self.nb_cols + 1
        if self.board[case1] == self.board[case2] and self.board[case1] == self.board[case3] and self.board[case1] == \
                self.board[case4] and self.board[case1] != 0:
            return self.board[case1], True
        case1 -= self.nb_cols
        case2 -= self.nb_cols
        case3 -= self.nb_cols
        case4 -= self.nb_cols
        if self.board[case1] == self.board[case2] and self.board[case1] == self.board[case3] and self.board[case1] == \
                self.board[case4] and self.board[case1] != 0:
            return self.board[case1], True
        # DIAGS RIGHT -> LEFT
        case1 = 27
        case2 = 33
        case3 = 39
        case4 = 45
        if self.board[case1] == self.board[case2] and self.board[case1] == self.board[case3]  and self.board[case1] == self.board[case4] and self.board[case1] != 0:
            return self.board[case1], True
        case1 -= self.nb_cols
        case2 -= self.nb_cols
        case3 -= self.nb_cols
        case4 -= self.nb_cols
        if self.board[case1] == self.board[case2] and self.board[case1] == self.board[case3]  and self.board[case1] == self.board[case4] and self.board[case1] != 0:
            return self.board[case1], True
        case1 += self.nb_cols - 1
        case2 += self.nb_cols - 1
        case3 += self.nb_cols - 1
        case4 += self.nb_cols - 1
        if self.board[case1] == self.board[case2] and self.board[case1] == self.board[case3]  and self.board[case1] == self.board[case4] and self.board[case1] != 0:
            return self.board[case1], True
        case1 -= 2*self.nb_cols - 1
        case2 -= 2*self.nb_cols - 1
        case3 -= 2*self.nb_cols - 1
        case4 -= 2*self.nb_cols - 1
        if self.board[case1] == self.board[case2] and self.board[case1] == self.board[case3]  and self.board[case1] == self.board[case4] and self.board[case1] != 0:
            return self.board[case1], True
        case1 += self.nb_cols - 1
        case2 += self.nb_cols - 1
        case3 += self.nb_cols - 1
        case4 += self.nb_cols - 1
        if self.board[case1] == self.board[case2] and self.board[case1] == self.board[case3]  and self.board[case1] == self.board[case4] and self.board[case1] != 0:
            return self.board[case1], True
        case1 += self.nb_cols - 1
        case2 += self.nb_cols - 1
        case3 += self.nb_cols - 1
        case4 += self.nb_cols - 1
        if self.board[case1] == self.board[case2] and self.board[case1] == self.board[case3]  and self.board[case1] == self.board[case4] and self.board[case1] != 0:
            return self.board[case1], True
        case1 -= 3*self.nb_cols - 2
        case2 -= 3*self.nb_cols - 2
        case3 -= 3*self.nb_cols - 2
        case4 -= 3*self.nb_cols - 2
        if self.board[case1] == self.board[case2] and self.board[case1] == self.board[case3]  and self.board[case1] == self.board[case4] and self.board[case1] != 0:
            return self.board[case1], True
        case1 += self.nb_cols - 1
        case2 += self.nb_cols - 1
        case3 += self.nb_cols - 1
        case4 += self.nb_cols - 1
        if self.board[case1] == self.board[case2] and self.board[case1] == self.board[case3]  and self.board[case1] == self.board[case4] and self.board[case1] != 0:
            return self.board[case1], True
        case1 += self.nb_cols - 1
        case2 += self.nb_cols - 1
        case3 += self.nb_cols - 1
        case4 += self.nb_cols - 1
        if self.board[case1] == self.board[case2] and self.board[case1] == self.board[case3]  and self.board[case1] == self.board[case4] and self.board[case1] != 0:
            return self.board[case1], True
        case1 += self.nb_cols - 1
        case2 += self.nb_cols - 1
        case3 += self.nb_cols - 1
        case4 += self.nb_cols - 1
        if self.board[case1] == self.board[case2] and self.board[case1] == self.board[case3]  and self.board[case1] == self.board[case4] and self.board[case1] != 0:
            return self.board[case1], True
        case1 -= 3*self.nb_cols - 2
        case2 -= 3*self.nb_cols - 2
        case3 -= 3*self.nb_cols - 2
        case4 -= 3*self.nb_cols - 2
        if self.board[case1] == self.board[case2] and self.board[case1] == self.board[case3]  and self.board[case1] == self.board[case4] and self.board[case1] != 0:
            return self.board[case1], True
        case1 += self.nb_cols - 1
        case2 += self.nb_cols - 1
        case3 += self.nb_cols - 1
        case4 += self.nb_cols - 1
        if self.board[case1] == self.board[case2] and self.board[case1] == self.board[case3]  and self.board[case1] == self.board[case4] and self.board[case1] != 0:
            return self.board[case1], True
        case1 += self.nb_cols - 1
        case2 += self.nb_cols - 1
        case3 += self.nb_cols - 1
        case4 += self.nb_cols - 1
        if self.board[case1] == self.board[case2] and self.board[case1] == self.board[case3]  and self.board[case1] == self.board[case4] and self.board[case1] != 0:
            return self.board[case1], True
        case1 -= 2*self.nb_cols - 1
        case2 -= 2*self.nb_cols - 1
        case3 -= 2*self.nb_cols - 1
        case4 -= 2*self.nb_cols - 1
        if self.board[case1] == self.board[case2] and self.board[case1] == self.board[case3]  and self.board[case1] == self.board[case4] and self.board[case1] != 0:
            return self.board[case1], True
        case1 += self.nb_cols - 1
        case2 += self.nb_cols - 1
        case3 += self.nb_cols - 1
        case4 += self.nb_cols - 1
        if self.board[case1] == self.board[case2] and self.board[case1] == self.board[case3]  and self.board[case1] == self.board[case4] and self.board[case1] != 0:
            return self.board[case1], True
        case1 -= self.nb_cols
        case2 -= self.nb_cols
        case3 -= self.nb_cols
        case4 -= self.nb_cols
        if self.board[case1] == self.board[case2] and self.board[case1] == self.board[case3]  and self.board[case1] == self.board[case4] and self.board[case1] != 0:
            return self.board[case1], True

        return 0.0, False

    def get_player_count(self) -> int:
        return 2

    def get_current_player_id(self) -> int:
        return self.current_player

    def get_information_state_for_player(self, player_id: int) -> 'InformationState':
        return Power4InformationState(self.current_player,
                                         self.board.copy())

    def get_available_actions_id_for_player(self, player_id: int) -> 'Iterable(int)':
        available_positions = []
        for i in range(self.nb_cols):
            for j in range(self.nb_lines):
                if self.board[j*self.nb_cols+i] == 0:
                    available_positions.append(j*self.nb_cols+i)
                    break
        return available_positions

    def __str__(self):
        gb = ""
        for i in range(7):
            for j in range(7):
                if self.board[j + 7 * (7-(i+1))] == 0:
                    gb += "_"
                elif self.board[j + 7 * (7-(i+1))] == 1:
                    gb += "X"
                else:
                    gb += "0"


            gb += "\n"
        return gb

    def copy_game_state(self):
        gs = Power4GameState()
        gs.board = self.board.copy()
        gs.current_player = self.current_player
        return gs


if __name__ == "__main__":
    gs = Power4GameState()
    gs.help()
    print(gs.get_available_actions_id_for_player(gs.get_current_player_id()))
    print(gs)
    print(gs.step(0,6))
    print(gs.get_available_actions_id_for_player(gs.get_current_player_id()))
    print(gs)
    print(gs.step(1,5))
    print(gs.get_available_actions_id_for_player(gs.get_current_player_id()))
    print(gs)
    print(gs.step(0,12))
    print(gs)
    print(gs.step(1,4))
    print(gs)
    print(gs.step(0,3))
    print(gs)
    print(gs.step(1,11))
    print(gs)
    print(gs.step(0,18))
    print(gs)
    print(gs.step(1,10))
    print(gs)
    print(gs.step(0,17))
    print(gs)
    print(gs.step(1,2))
    print(gs)
    print(gs.step(0,24))
    print(gs)
    print(gs.get_available_actions_id_for_player(gs.get_current_player_id()))
    print(gs)