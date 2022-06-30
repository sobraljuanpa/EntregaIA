from ast import For
from math import inf
from operator import truediv
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

  def es_terminal(self, tablero:GameBoard):
    if tablero.get_available_moves().__len__() == 0:
      return True

  def minimax(self, tablero:GameBoard, profundidad:int, esMax:bool):
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
        auxTablero.grid[posicion[0]][posicion[1]] = 2 # no contemplo un 4
        valor = self.minimax(auxTablero, profundidad - 1, not esMax)
        best = min(best, valor)
      return best

  def get_possible_states(self, board:GameBoard, move:bool):
    """
      Based on the current board, return a list with the possible resulting states
    """
    
    if move:
      states = [board] * board.get_available_moves()
    else:
      states = [board] * board.get_available_cells() # no contemplo posibilidad de que haya 4s insertados
    

    

    return states

  def play(self, board:GameBoard):
    # Caso base, nodo hoja, mejor siguiente jugada
    values = [-1] * 4                                       # inicializo lista para almacenar los valores de las jugadas

    for moveIndex in board.get_available_moves():           # para cada jugada posible
      auxBoard = board.clone()                              # clono tablero para simular jugada
      lost = auxBoard.move(moveIndex)                       # simulo jugada
      values[moveIndex] += self.count_adjacencies(auxBoard, moveIndex)
      values[moveIndex] += self.minimax(auxBoard, 5, True)

    print('Valor movimientos:')
    for i in range(4):
      print(self.int_to_string[i], ': ', values[i])
    print('Valor np.argmax de values: {}'.format(np.argmax(values)))
    return np.argmax(values)

  def heuristic_utility(self, board: GameBoard):
    return board.get_available_cells().__len__()# agregar sumatoria de todas las posibles uniones despues de justificar en cada direccion
