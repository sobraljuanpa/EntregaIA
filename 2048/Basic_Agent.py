from ast import For
from math import inf
from operator import truediv
import random
# from telnetlib import GA
# from tkinter import W
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


class BasicAgent(Agent):

    int_to_string = ['UP', 'DOWN', 'LEFT', 'RIGHT']

    def init(self):
        pass

    def es_terminal(self, tablero: GameBoard):
        if tablero.get_available_moves().__len__() == 0:
            return True

    # Decidimos utilizar una funcion randomica para evaluar si el numero es 2 o 4 ya que la posibilidad de que sea un 4 es muchisimo menor que la de un 2 y estariamos calculando
    # movimientos que en terminos de numeros serian mejores pero no serian posibles de hacer.
    def randomNumberToInsert(self):
        if np.random.random_integers(0, 99) < 90:
            return 2
        else:
            return 4

    def minimax(self, tablero: GameBoard, profundidad: int, esMax: bool):
        if self.es_terminal(tablero) or profundidad == 0:
            return self.heuristic_utility(tablero)

        if esMax:
            best = -inf
            for movida in tablero.get_available_moves():
                auxTablero = tablero.clone()
                auxTablero.move(movida)
                valor = self.minimax(auxTablero, profundidad - 1, not esMax)
                best = max(best, valor)
            return best
        else:
            best = inf
            for posicion in tablero.get_available_cells():
                auxTablero = tablero.clone()
                # no contemplo un 4
                auxTablero.grid[posicion[0]][posicion[1]] = self.randomNumberToInsert()
                valor = self.minimax(auxTablero, profundidad - 1, not esMax)
                best = min(best, valor)
            return best

    def minimaxAb(self, tablero: GameBoard, profundidad: int, esMax: bool, alpha: int, beta: int):
        if self.es_terminal(tablero) or profundidad == 0:
            return self.heuristic_utility(tablero)

        if esMax:
            best = -inf
            for movida in tablero.get_available_moves():
                auxTablero = tablero.clone()
                auxTablero.move(movida)
                valor = self.minimaxAb(
                    auxTablero, profundidad - 1, False, alpha, beta)
                best = max(best, valor)
                if best >= beta:
                    return best
                alpha = max(alpha, best)
            return best
        else:
            best = inf
            for posicion in tablero.get_available_cells():
                auxTablero = tablero.clone()

                auxTablero.grid[posicion[0]][posicion[1]] = self.randomNumberToInsert()
                valor = self.minimaxAb(auxTablero, profundidad - 1, True, alpha, beta)
                best = min(best, valor)
                if best <= alpha:
                    return best
                beta = min(beta, best)
            return best

    def isBorder (self, row, column):
        return (row == 0 or row == 3) and (column == 0 or column == 3)

    def play(self, board: GameBoard):
        # Caso base, nodo hoja, mejor siguiente jugada
        # inicializo lista para almacenar los valores de las jugadas
        values = [-1] * 4

        availableMoves = board.get_available_moves()
        # print('Available moves ' + str(availableMoves))
        for moveIndex in availableMoves:           # para cada jugada posible
            # clono tablero para simular jugada
            auxBoard = board.clone()
            lost = auxBoard.move(moveIndex)                       # simulo jugada

            # Evaluo el primer movimiento, Seguro se puede mejorar la logica pero ahora no me da.
            values[moveIndex] += self.minimaxAb(auxBoard, 0, True, -np.inf, np.inf) # Chancho pero sirve por ahora
            values[moveIndex] += self.minimaxAb(auxBoard, 1, True, -np.inf, np.inf)

        if np.amax(values) >= 0:
            max = np.argmax(values)
        else:
            max = self.obtainMax(values)

        return max

    def obtainMax(self, values):
        max = -inf
        for i in range(len(values)):
            if values[i] != -1 and values[i] >= max:
              max = i

        return max

    def emptyTitles(self, board: GameBoard):
        return board.get_available_cells().__len__()

    def max_title_position(self, tablero: GameBoard):
        max_title = tablero.get_max_tile()
        if tablero.grid[0][0] == max_title:
            return 500000
        else:
            return -500000

    def weighted_board(self, board: GameBoard):
        result = 0
        tablero = board.clone()
        for i in range(len(tablero.grid)):
            for j in range(len(tablero.grid)):
                result += tablero.grid[i][j] * WEIGHT_MATRIX[i][j]

        return result
        
    def smoothness(self, tablero: GameBoard):
        auxTablero = tablero.clone()
        smothness = 0
        availableMoves = tablero.get_available_moves()
        if len(availableMoves) > 0:
            for r in auxTablero.grid:  # Me guardo fila.
                for i in range(2):
                    smothness -= abs(r[i] - r[i + 1])
                    pass
            for j in range(3):
                for k in range(2):
                    smothness -= abs(auxTablero.grid[k]
                                     [j] - auxTablero.grid[k+1][j])

        return smothness

    def monotonicity(self, board: GameBoard):
        mono = 0
        tablero = board.clone()
        # Left/right
        for r in tablero.grid:
            diff = r[0] - r[1]
            for i in range(2):
                if (r[i] - r[i+1]) * diff <= 0:
                    mono += 1
                diff = r[i] - r[i+1]

        # Up/down
        for j in range(3):
            diff = tablero.grid[0][j] - tablero.grid[1][j]
            for k in range(2):
                if (tablero.grid[k][j] - tablero.grid[k+1][j]) * diff <= 0:
                    mono += 1
                diff = tablero.grid[k][j] - tablero.grid[k+1][j]

        return mono

    def talesValues(self, board: GameBoard):
        totalValue = 0
        for i in range(3):
            for j in range(3):
                totalValue += board.grid[i][j] ** 2
        
        return totalValue

    def heuristic_utility(self, board: GameBoard):
        empty_titles = self.emptyTitles(board) * 340
        smoothness = self.smoothness(board) * 100
        max_title = self.max_title_position(board) * 55
        weighted_board = self.weighted_board(board)
        mono = self.monotonicity(board) * 18
        totalValue = self.talesValues(board)
        return empty_titles + max_title + weighted_board + totalValue + mono
