from ast import For
from Agent import Agent
from GameBoard import GameBoard
import numpy as np


class BasicAgent(Agent):

  int_to_string = ['UP', 'DOWN', 'LEFT', 'RIGHT']

  def init(self):
    pass

  def count_adjacencies(self, board:GameBoard, direction:int):
    count = 0
    auxBoard = board.clone()
    if direction == 0:
      auxBoard.justify_up()
      for aux in range(3):
        for i in [1,2,3]:
          for j in range(4):
            if auxBoard.grid[i - 1][j] == auxBoard.grid[i][j] and auxBoard.grid[i][j] !=0:
              count += 1
    if direction == 1:
      auxBoard.justify_down()
      for aux in range(3):
        for i in range(3):
          for j in range(4):
            if auxBoard.grid[i + 1][j] == auxBoard.grid[i][j] and auxBoard.grid[i][j] !=0:
              count += 1
    if direction == 2:
      auxBoard.justify_left()
      for aux in range(3):
        for i in range(4):
          for j in [1,2,3]:
            if auxBoard.grid[i][j - 1] == auxBoard.grid[i][j] and auxBoard.grid[i][j] !=0:
              count += 1
    if direction == 3:
      auxBoard.justify_right()
      for aux in range(3):
        for i in range(4):
          for j in range(3):
            if auxBoard.grid[i][j + 1] == auxBoard.grid[i][j] and auxBoard.grid[i][j] !=0:
              count += 1

    return count

  def play(self, board:GameBoard):
    # Caso base, nodo hoja, mejor siguiente jugada
    values = [-1] * 4                                       # inicializo lista para almacenar los valores de las jugadas

    for moveIndex in board.get_available_moves():           # para cada jugada posible
      auxBoard = board.clone()                              # clono tablero para simular jugada
      lost = auxBoard.move(moveIndex)                       # simulo jugada
      values[moveIndex] += self.count_adjacencies(board, moveIndex)
      values[moveIndex] += self.heuristic_utility(auxBoard)  # asigno valor de la heuristica a la posicion asociada a la jugada en la lista

    print('Valor movimientos:')
    for i in range(4):
      print(self.int_to_string[i], ': ', values[i])
    print('Valor np.argmax de values: {}'.format(np.argmax(values)))
    return np.argmax(values)

  def heuristic_utility(self, board: GameBoard):
    return board.get_available_cells().__len__() # se itera sobre heuristica inicial
