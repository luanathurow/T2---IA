# operacao/minimax.py

from tic_tac_toe import TicTacToe
import random


def _minimax(game: TicTacToe, is_maximizing: bool):
    """
    Minimax simples (retorna utilidade final):
    retorna 1 se X vencer, -1 se O vencer, 0 se empate.
    Avalia sempre do ponto de vista do X (1).
    """
    winner = game.checar_vencedor()
    if winner is not None:
        return winner

    if game.checar_empate():
        return 0

    if is_maximizing:
        best = -float('inf')
        for (i, j) in game.movimentos_disponiveis():
            game.board[i][j] = 1  # X
            val = _minimax(game, False)
            game.board[i][j] = 0
            if val > best:
                best = val
        return best
    else:
        best = float('inf')
        for (i, j) in game.movimentos_disponiveis():
            game.board[i][j] = -1  # O
            val = _minimax(game, True)
            game.board[i][j] = 0
            if val < best:
                best = val
        return best


def melhor_jogada(game: TicTacToe, jogador: int):
    """
    Retorna a melhor jogada (linha, coluna) para `jogador` usando Minimax.
    `jogador` deve ser 1 (X) ou -1 (O). Retorna None se não houver movimentos.
    """
    movs = game.movimentos_disponiveis()
    if not movs:
        return None

    if jogador == 1:
        best_val = -float('inf')
        best_move = None
        for (i, j) in movs:
            game.board[i][j] = 1
            val = _minimax(game, False)
            game.board[i][j] = 0
            if val > best_val:
                best_val = val
                best_move = (i, j)
        return best_move
    else:
        best_val = float('inf')
        best_move = None
        for (i, j) in movs:
            game.board[i][j] = -1
            val = _minimax(game, True)
            game.board[i][j] = 0
            if val < best_val:
                best_val = val
                best_move = (i, j)
        return best_move


def melhor_jogada_modo(game: TicTacToe, jogador: int, modo: str = 'dificil'):
    """
    Retorna a melhor jogada considerando o modo de dificuldade.
    - modo = 'medio'  : 50% Minimax, 50% aleatório
    - modo = 'dificil': sempre Minimax (padrão)
    """
    modo = modo.lower() if modo else 'dificil'

    if modo == 'medio':
        # 50% das vezes usa Minimax, 50% aleatório
        if random.random() < 0.5:
            return melhor_jogada(game, jogador)
        else:
            movs = game.movimentos_disponiveis()
            if not movs:
                return None
            return random.choice(movs)

    # padrão: difícil (sempre Minimax)
    return melhor_jogada(game, jogador)
