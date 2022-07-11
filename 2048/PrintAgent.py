from argparse import HelpFormatter
from ast import For
from math import inf
from operator import truediv
import random
from Agent import Agent
from GameBoard import GameBoard
import numpy as np

MAX_TITLE_CREDIT = 10e4
WEIGHT_MATRIX = [
    [4**15, 4**14, 4**13, 4**12],
    [4**8, 4**9, 4**10, 4**11],
    [4**7, 4**6, 4**5, 4**4],
    [4**0, 4**1, 4**2, 4**3]
]

class PrintAgent(Agent):

    int_to_string = ['UP', 'DOWN', 'LEFT', 'RIGHT']

    def init(self):
        pass

    def is_terminal(self, tablero: GameBoard):
        if tablero.get_available_moves().__len__() == 0:
            return True

    def randomNumberToInsert(self):
        if np.random.random_integers(0, 99) < 90:
            return 2
        else:
            return 4

    def minimaxAb(self, board: GameBoard, depth: int, esMax: bool, alpha: int, beta: int):
        if self.is_terminal(board) or depth == 0:
            return self.heuristic_utility(board)

        if esMax:
            best = -inf
            for move in board.get_available_moves():
                auxBoard = board.clone()
                auxBoard.move(move)
                valor = self.minimaxAb(
                    auxBoard, depth - 1, False, alpha, beta)
                best = max(best, valor)
                if best >= beta:
                    return best
                alpha = max(alpha, best)
            return best
        else:
            best = inf
            for position in board.get_available_cells():
                auxBoard = board.clone()
                auxBoard.grid[position[0]][position[1]] = self.randomNumberToInsert()
                valor = self.minimaxAb(auxBoard, depth - 1, True, alpha, beta)
                best = min(best, valor)
                if best <= alpha:
                    return best
                beta = min(beta, best)
            return best

    def play(self, board: GameBoard):
        values = [-1] * 4

        availableMoves = board.get_available_moves()
        print('Available moves ' + str(availableMoves))
        for moveIndex in availableMoves:
            auxBoard = board.clone()
            _move = auxBoard.move(moveIndex)

            # We evaluate the move previously done.
            values[moveIndex] += self.minimaxAb(auxBoard, 0, True, -np.inf, np.inf)
            values[moveIndex] += self.minimaxAb(auxBoard, 2, True, -np.inf, np.inf)

        print('Valor movimientos:')
        for i in range(4):
            print(self.int_to_string[i], ': ', values[i])

        if np.amax(values) >= 0:
            max = np.argmax(values)
        else:
            max = self.obtainMax(values)
        print('Valor np.argmax de values: {}'.format(max))

        return max

    def obtainMax(self, values):
        max = -inf
        for i in range(len(values)):
            if values[i] != -1 and values[i] >= max:
              max = i

        return max

    def emptyTiles(self, board: GameBoard):
        return board.get_available_cells().__len__()

    def max_tile_position(self, board: GameBoard):
        max_tile = board.get_max_tile()
        if board.grid[0][0] == max_tile:
            return 500000
        else:
            return -500000

    def weighted_board(self, board: GameBoard):
        result = 0
        auxBoard = board.clone()
        for i in range(len(auxBoard.grid)):
            for j in range(len(auxBoard.grid)):
                result += auxBoard.grid[i][j] * WEIGHT_MATRIX[i][j]

        return result
        
    def smoothness(self, board: GameBoard):
        auxBoard = board.clone()
        smothness = 0
        availableMoves = board.get_available_moves()
        if len(availableMoves) > 0:
            for r in auxBoard.grid:
                for i in range(2):
                    smothness -= abs(r[i] - r[i + 1])
                    pass
            for j in range(3):
                for k in range(2):
                    smothness -= abs(auxBoard.grid[k]
                                        [j] - auxBoard.grid[k+1][j])

        return smothness

    def monotonicity(self, board: GameBoard):
        mono = 0
        auxBoard = board.clone()
        # Left/right
        for r in auxBoard.grid:
            diff = r[0] - r[1]
            for i in range(2):
                if (r[i] - r[i+1]) * diff <= 0:
                    mono += 1
                diff = r[i] - r[i+1]

        # Up/down
        for j in range(3):
            diff = auxBoard.grid[0][j] - auxBoard.grid[1][j]
            for k in range(2):
                if (auxBoard.grid[k][j] - auxBoard.grid[k+1][j]) * diff <= 0:
                    mono += 1
                diff = auxBoard.grid[k][j] - auxBoard.grid[k+1][j]

        return mono

    def tilesValues(self, board: GameBoard):
        totalValue = 0
        for i in range(3):
            for j in range(3):
                totalValue += board.grid[i][j] ** 2
        
        return totalValue

    def heuristic_utility(self, board: GameBoard):
        empty_titles = self.emptyTiles(board) * 340
        smoothness = self.smoothness(board) * 5
        max_title = self.max_tile_position(board) * 55
        weighted_board = self.weighted_board(board)
        mono = self.monotonicity(board) * 18
        totalValue = self.tilesValues(board)
        return empty_titles + max_title + weighted_board + totalValue + mono + smoothness