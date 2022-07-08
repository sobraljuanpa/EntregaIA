from argparse import HelpFormatter
from ast import For
from math import inf
from operator import truediv
import random
# from telnetlib import GA
# from tkinter import W
from Agent import Agent
from GameBoard import GameBoard
import numpy as np
import Helper

class PrintAgent(Agent):

    int_to_string = ['UP', 'DOWN', 'LEFT', 'RIGHT']

    def init(self):
        pass

    def es_terminal(self, tablero: GameBoard):
        if tablero.get_available_moves().__len__() == 0:
            return True

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
                auxTablero.grid[posicion[0]][posicion[1]] = Helper.randomNumberToInsert()
                valor = self.minimax(auxTablero, profundidad - 1, not esMax)
                best = min(best, valor)
            return best

    def minimaxAb(self, tablero: GameBoard, profundidad: int, esMax: bool, alpha: int, beta: int):
        if self.es_terminal(tablero) or profundidad == 0:
            return Helper.heuristic_utility(tablero)

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

                auxTablero.grid[posicion[0]][posicion[1]] = Helper.randomNumberToInsert()
                valor = self.minimaxAb(auxTablero, profundidad - 1, True, alpha, beta)
                best = min(best, valor)
                if best <= alpha:
                    return best
                beta = min(beta, best)
            return best

    def play(self, board: GameBoard):
        # Caso base, nodo hoja, mejor siguiente jugada
        # inicializo lista para almacenar los valores de las jugadas
        values = [-1] * 4

        availableMoves = board.get_available_moves()
        print('Available moves ' + str(availableMoves))
        for moveIndex in availableMoves:           # para cada jugada posible
            # clono tablero para simular jugada
            auxBoard = board.clone()
            lost = auxBoard.move(moveIndex) # simulo jugada

            # Evaluo el primer movimiento, Seguro se puede mejorar la logica pero ahora no me da.
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