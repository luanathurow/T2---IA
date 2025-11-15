class TicTacToe:
    def __init__(self):
        # 1 = X (rede neural)
        # -1 = O (minimax ou humano)
        # 0 = vazio
        self.board = [[0 for _ in range(3)] for _ in range(3)]

    def mostrar(self):
        symbols = {1: 'X', -1: 'O', 0: ' '}
        print("\nTabuleiro:")
        for linha in self.board:
            print("|".join(symbols[c] for c in linha))
            print("-" * 5)

    def jogada_valida(self, linha, col):
        return self.board[linha][col] == 0

    def fazer_jogada(self, linha, col, jogador):
        """
        jogador = 1 (X) ou -1 (O)
        """
        if self.jogada_valida(linha, col):
            self.board[linha][col] = jogador
            return True
        return False

    def checar_vencedor(self):
        linhas = self.board
        colunas = list(zip(*self.board))
        diagonais = [
            [self.board[i][i] for i in range(3)],
            [self.board[i][2-i] for i in range(3)]
        ]

        # Verifica linha, coluna ou diagonal completa
        for linha in linhas + colunas + diagonais:
            if linha.count(linha[0]) == 3 and linha[0] != 0:
                return linha[0]  # retorna 1 (X) ou -1 (O)

        return None  # sem vencedor ainda

    def checar_empate(self):
        if any(0 in linha for linha in self.board):
            return False  # ainda tem jogada
        return self.checar_vencedor() is None

    def jogo_terminou(self):
        if self.checar_vencedor() is not None:
            return True
        if self.checar_empate():
            return True
        return False

    def movimentos_disponiveis(self):
        movs = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    movs.append((i, j))
        return movs

    def reset(self):
        self.board = [[0 for _ in range(3)] for _ in range(3)]
