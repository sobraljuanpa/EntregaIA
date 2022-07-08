from GameBoard import GameBoard
import numpy as np

WEIGHT_MATRIX = [
    [4**15, 4**14, 4**13, 4**12],
    [4**8, 4**9, 4**10, 4**11],
    [4**7, 4**6, 4**5, 4**4],
    [4**0, 4**1, 4**2, 4**3]
]


class Helper:
    def init(self):
        pass

    def randomNumberToInsert(self):
        if np.random.random_integers(0, 99) < 90:
            return 2
        else:
            return 4

    def emptyTitles(self, board: GameBoard):
        return board.get_available_cells().__len__()

    def max_title_position(tablero: GameBoard):
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