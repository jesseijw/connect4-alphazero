# Connect 4 game engine - board logic, moves, win detection

import numpy as np

# board is 6 rows x 7 cols
# player 1 = 1, player 2 = -1, empty = 0

ROWS = 6
COLS = 7

class Connect4:

    def __init__(self):
        # start with empty board
        self.board = np.zeros((ROWS, COLS), dtype=int)
        self.current_player = 1
        self.winner = None
        self.game_over = False

    def get_valid_moves(self):
        # a column is valid if the top row is still empty
        validMoves = []
        for col in range(COLS):
            if self.board[0][col] == 0:
                validMoves.append(col)
        return validMoves

    def drop_piece(self, col):
        # drop from the bottom up, find lowest empty row
        if col not in self.get_valid_moves():
            return False

        for row in range(ROWS - 1, -1, -1):
            if self.board[row][col] == 0:
                self.board[row][col] = self.current_player
                break

        # check if that move won the game
        if self.check_win(self.current_player):
            self.winner = self.current_player
            self.game_over = True
        elif len(self.get_valid_moves()) == 0:
            # no moves left = draw
            self.game_over = True

        # switch players
        self.current_player *= -1
        return True

    def check_win(self, player):
        # check horizontal
        for row in range(ROWS):
            for col in range(COLS - 3):
                if all(self.board[row][col + i] == player for i in range(4)):
                    return True

        # check vertical
        for row in range(ROWS - 3):
            for col in range(COLS):
                if all(self.board[row + i][col] == player for i in range(4)):
                    return True

        # check diagonal going down-right
        for row in range(ROWS - 3):
            for col in range(COLS - 3):
                if all(self.board[row + i][col + i] == player for i in range(4)):
                    return True

        # check diagonal going down-left
        for row in range(ROWS - 3):
            for col in range(3, COLS):
                if all(self.board[row + i][col - i] == player for i in range(4)):
                    return True

        return False

    def clone(self):
        # make a copy of the game so mcts can simulate without messing up real game
        newGame = Connect4()
        newGame.board = self.board.copy()
        newGame.current_player = self.current_player
        newGame.winner = self.winner
        newGame.game_over = self.game_over
        return newGame

    def get_board_state(self):
        # return board from perspective of current player
        # this is what gets fed into the neural net
        return self.board * self.current_player

    def __str__(self):
        # just for printing in terminal
        symbols = {0: ".", 1: "X", -1: "O"}
        rows = []
        for row in self.board:
            rows.append(" ".join(symbols[val] for val in row))
        return "\n".join(rows)
