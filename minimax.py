# operacao/minimax.py

from tic_tac_toe import TicTacToe
import random


def _minimax_ab(game: TicTacToe, is_maximizing: bool, alpha: float, beta: float, cache: dict):
    """
    Minimax com poda alpha-beta e tabela de transposição (cache).
    Retorna utilidade final: 1 se X vencer, -1 se O vencer, 0 se empate.
    Avalia do ponto de vista de X (1).
    """
    # hash do estado simples: tuple do tabuleiro plano + vez
    board_key = tuple(game.board[i][j] for i in range(3) for j in range(3))
    key = (board_key, is_maximizing)
    if key in cache:
        return cache[key]

    winner = game.checar_vencedor()
    if winner is not None:
        cache[key] = winner
        return winner

    if game.checar_empate():
        cache[key] = 0
        return 0

    if is_maximizing:
        value = -float('inf')
        for (i, j) in game.movimentos_disponiveis():
            game.board[i][j] = 1  # X
            val = _minimax_ab(game, False, alpha, beta, cache)
            game.board[i][j] = 0
            if val > value:
                value = val
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # poda beta
        cache[key] = value
        return value
    else:
        value = float('inf')
        for (i, j) in game.movimentos_disponiveis():
            game.board[i][j] = -1  # O
            val = _minimax_ab(game, True, alpha, beta, cache)
            game.board[i][j] = 0
            if val < value:
                value = val
            beta = min(beta, value)
            if alpha >= beta:
                break  # poda alpha
        cache[key] = value
        return value


def melhor_jogada(game: TicTacToe, jogador: int):
    """
    Retorna a melhor jogada (linha, coluna) para `jogador` usando Minimax.
    `jogador` deve ser 1 (X) ou -1 (O). Retorna None se não houver movimentos.
    """
    movs = game.movimentos_disponiveis()
    if not movs:
        return None

    # cria cache por chamada para evitar poluir entre jogos
    cache = {}
    if jogador == 1:
        best_val = -float('inf')
        best_move = None
        for (i, j) in movs:
            game.board[i][j] = 1
            val = _minimax_ab(game, False, -float('inf'), float('inf'), cache)
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
            val = _minimax_ab(game, True, -float('inf'), float('inf'), cache)
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
