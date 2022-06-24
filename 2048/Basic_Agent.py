from ast import For
from Agent import Agent
from GameBoard import GameBoard
import numpy as np


class BasicAgent(Agent):

  int_to_string = ['UP', 'DOWN', 'LEFT', 'RIGHT']

  def init(self):
    pass

  def play(self, board:GameBoard):
    # Caso base, nodo hoja, mejor siguiente jugada
    values = [-1] * 4                                       # inicializo lista para almacenar los valores de las jugadas

    for moveIndex in board.get_available_moves():           # para cada jugada posible
      auxBoard = board.clone()                              # clono tablero para simular jugada
      lost = auxBoard.move(moveIndex)                       # simulo jugada
      values[moveIndex] = self.heuristic_utility(auxBoard)  # asigno valor de la heuristica a la posicion asociada a la jugada en la lista

    print('Valor movimientos:')
    for i in range(4):
      print(self.int_to_string[i], ': ', values[i])
    print('Valor np.argmax de values: {}'.format(np.argmax(values)))
    return np.argmax(values)

  def heuristic_utility(self, board: GameBoard):
    return board.get_available_cells().__len__()
