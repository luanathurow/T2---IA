from tic_tac_toe import TicTacToe
from minimax import melhor_jogada


def run():
    jogo = TicTacToe()
    jogador = 1
    while not jogo.jogo_terminou():
        mv = melhor_jogada(jogo, jogador)
        if mv is None:
            break
        jogo.fazer_jogada(mv[0], mv[1], jogador)
        jogador *= -1

    print('Final board:')
    for row in jogo.board:
        print(row)
    print('winner=', jogo.checar_vencedor(), 'draw=', jogo.checar_empate())


if __name__ == '__main__':
    run()
