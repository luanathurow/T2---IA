import argparse
import random
import numpy as np

from tic_tac_toe import TicTacToe
from rede_neural import NeuralNetwork
from minimax import melhor_jogada_modo


def play_game(rede, modo_minimax: str, network_is_X: bool):
    jogo = TicTacToe()
    player = 1  # X começa sempre

    while not jogo.jogo_terminou():
        movs = jogo.movimentos_disponiveis()
        if player == 1:
            if network_is_X:
                mv = rede.escolher_jogada(jogo.board, movs)
                if not jogo.jogada_valida(mv[0], mv[1]):
                    mv = random.choice(movs)
                jogo.fazer_jogada(mv[0], mv[1], 1)
            else:
                mv = melhor_jogada_modo(jogo, 1, modo=modo_minimax)
                if mv is None:
                    mv = random.choice(movs)
                jogo.fazer_jogada(mv[0], mv[1], 1)
        else:
            if not network_is_X:
                mv = rede.escolher_jogada(jogo.board, movs)
                if not jogo.jogada_valida(mv[0], mv[1]):
                    mv = random.choice(movs)
                jogo.fazer_jogada(mv[0], mv[1], -1)
            else:
                mv = melhor_jogada_modo(jogo, -1, modo=modo_minimax)
                if mv is None:
                    mv = random.choice(movs)
                jogo.fazer_jogada(mv[0], mv[1], -1)

        player *= -1

    winner = jogo.checar_vencedor()
    if winner == 1:
        return 'rede' if network_is_X else 'minimax'
    elif winner == -1:
        return 'rede' if not network_is_X else 'minimax'
    else:
        return 'empate'


def evaluate(rede, n_games=100, modo_minimax='dificil'):
    results = {'rede': 0, 'minimax': 0, 'empate': 0}

    # alterna quem começa para ser justo
    network_starts = True
    for i in range(n_games):
        res = play_game(rede, modo_minimax, network_is_X=network_starts)
        results[res] += 1
        network_starts = not network_starts

    return results


def main():
    parser = argparse.ArgumentParser(description='Avalia Rede vs Minimax')
    parser.add_argument('--games', '-g', type=int, default=100, help='Número de partidas')
    parser.add_argument('--modo', '-m', type=str, default='dificil', choices=['medio', 'dificil'], help='Modo do Minimax')
    parser.add_argument('--chrom', '-c', type=str, default='best_chromosome.npy', help='Arquivo .npy com cromossomo (opcional)')
    parser.add_argument('--hidden', type=int, default=18, help='Tamanho da camada oculta (se carregar cromossomo)')

    args = parser.parse_args()

    # Tenta carregar cromossomo; se falhar, usa rede aleatória
    rede = None
    try:
        chrom = np.load(args.chrom)
        rede = NeuralNetwork.from_chromosome(chrom, hidden_size=args.hidden)
        print(f"Carregado cromossomo de '{args.chrom}'")
    except Exception as e:
        print(f"Não foi possível carregar '{args.chrom}', usando rede aleatória. Detalhe: {e}")
        rede = NeuralNetwork(hidden_size=args.hidden)

    print(f"Executando {args.games} partidas (modo minimax={args.modo})...")
    stats = evaluate(rede, n_games=args.games, modo_minimax=args.modo)

    total = args.games
    print('\nResultados:')
    print(f"Rede venceu: {stats['rede']} ({stats['rede']/total*100:.2f}%)")
    print(f"Minimax venceu: {stats['minimax']} ({stats['minimax']/total*100:.2f}%)")
    print(f"Empates: {stats['empate']} ({stats['empate']/total*100:.2f}%)")


if __name__ == '__main__':
    main()
