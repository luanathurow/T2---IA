from tic_tac_toe import TicTacToe
from minimax import melhor_jogada_modo


def print_board(game):
    symbols = {1: 'X', -1: 'O', 0: ' '}
    for row in game.board:
        print('|'.join(symbols[c] for c in row))
    print('-----')


def reproduce():
    # Sequência que leva ao tabuleiro mostrado pelo usuário:
    # Topo: O O O
    # Meio: X X  
    # Baixo: X    
    # Supondo humano = X (1) começa, e Minimax = O (-1)
    seq_human = [(1, 0), (1, 1), (2, 0)]  # 3 jogadas do humano

    jogo = TicTacToe()
    jogador = 1  # humano começa como X

    for move in seq_human:
        # humano joga a jogada fixa
        l, c = move
        ok = jogo.fazer_jogada(l, c, 1)
        print(f"Humano joga {(l,c)} -> {'ok' if ok else 'INVÁLIDO'}")
        print_board(jogo)
        if jogo.checar_vencedor() is not None:
            print('Vencedor detectado após jogada humana:', jogo.checar_vencedor())
            break

        # Minimax joga uma resposta (modo difícil)
        mv = melhor_jogada_modo(jogo, -1, modo='dificil')
        print('Minimax escolheu:', mv)
        if mv is not None:
            jogo.fazer_jogada(mv[0], mv[1], -1)
        print_board(jogo)
        if jogo.checar_vencedor() is not None:
            print('Vencedor detectado após Minimax:', jogo.checar_vencedor())
            break

    print('Estado final:')
    print_board(jogo)
    print('checar_vencedor() ->', jogo.checar_vencedor())


if __name__ == '__main__':
    reproduce()
