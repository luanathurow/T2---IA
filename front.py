from tic_tac_toe import TicTacToe
import os

def limpar_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def exibir_menu():
    print("\n===== JOGO DA VELHA - IA =====")
    print("1 - Jogar contra outro humano")
    print("2 - Jogar contra máquina (aleatória por enquanto)")
    print("3 - (DEPOIS) Jogar contra Minimax")
    print("4 - (DEPOIS) Jogar contra Rede Neural Treinada")
    print("0 - Sair")
    return input("Escolha uma opção: ")

# ----------------------------------------------------------------------
# Jogada da máquina ALEATÓRIA (temporária)
# ----------------------------------------------------------------------
import random
def jogada_maquina_aleatoria(jogo: TicTacToe, jogador):
    movimentos = jogo.movimentos_disponiveis()
    l, c = random.choice(movimentos)
    jogo.fazer_jogada(l, c, jogador)

# ----------------------------------------------------------------------
# Jogada do humano
# ----------------------------------------------------------------------
def jogada_humano(jogo: TicTacToe, jogador):
    while True:
        try:
            pos = input("Digite linha e coluna (ex: 1 2): ")
            l, c = map(int, pos.split())
            if l in range(3) and c in range(3):
                if jogo.fazer_jogada(l, c, jogador):
                    return
                else:
                    print("Casa ocupada! Tente de novo.")
            else:
                print("Posição inválida! Use valores entre 0 e 2.")
        except Exception:
            print("Entrada inválida!")


# ----------------------------------------------------------------------
# Jogada do Minimax (com modos)
# ----------------------------------------------------------------------
def jogada_minimax(jogo: TicTacToe, jogador, modo: str = 'dificil'):
    from operacao.minimax import melhor_jogada_modo

    move = melhor_jogada_modo(jogo, jogador, modo)
    if move is None:
        # fallback para aleatória (não deve ocorrer)
        jogada_maquina_aleatoria(jogo, jogador)
    else:
        l, c = move
        jogo.fazer_jogada(l, c, jogador)

# ----------------------------------------------------------------------
# Lógica completa de uma partida
# ----------------------------------------------------------------------
def jogar_contra_maquina():
    jogo = TicTacToe()
    jogador = 1  # começa sempre o X (IA no futuro)

    while True:
        limpar_console()
        jogo.mostrar()

        # Jogador humano
        if jogador == 1:
            print("Sua vez (X)")
            jogada_humano(jogo, 1)
        else:
            print("Vez da máquina (O)")
            jogada_maquina_aleatoria(jogo, -1)

        vencedor = jogo.checar_vencedor()
        if vencedor is not None:
            limpar_console()
            jogo.mostrar()
            if vencedor == 1:
                print("\nVocê venceu! 🎉")
            else:
                print("\nA máquina venceu! 🤖")
            break

        if jogo.checar_empate():
            limpar_console()
            jogo.mostrar()
            print("\nEmpate!")
            break

        jogador *= -1  # troca 1 → -1 → 1 → -1 ...

# ----------------------------------------------------------------------
# Lógica para jogar humano x humano (útil para testes)
# ----------------------------------------------------------------------
def jogar_humano_vs_humano():
    jogo = TicTacToe()
    jogador = 1

    while True:
        limpar_console()
        jogo.mostrar()
        print(f"Jogador { 'X' if jogador == 1 else 'O' }")

        jogada_humano(jogo, jogador)

        vencedor = jogo.checar_vencedor()
        if vencedor is not None:
            limpar_console()
            jogo.mostrar()
            print(f"\nJogador {'X' if vencedor == 1 else 'O'} venceu! 🎉")
            break

        if jogo.checar_empate():
            limpar_console()
            jogo.mostrar()
            print("\nEmpate!")
            break

        jogador *= -1


def jogar_contra_minimax(modo: str = 'dificil'):
    jogo = TicTacToe()
    jogador = 1  # começa sempre o X (humano)

    while True:
        limpar_console()
        jogo.mostrar()

        # Jogador humano
        if jogador == 1:
            print("Sua vez (X)")
            jogada_humano(jogo, 1)
        else:
            print(f"Vez do Minimax ({modo}) (O)")
            jogada_minimax(jogo, -1, modo)

        vencedor = jogo.checar_vencedor()
        if vencedor is not None:
            limpar_console()
            jogo.mostrar()
            if vencedor == 1:
                print("\nVocê venceu! 🎉")
            else:
                print("\nO Minimax venceu! 🤖")
            break

        if jogo.checar_empate():
            limpar_console()
            jogo.mostrar()
            print("\nEmpate!")
            break

        jogador *= -1  # troca 1 → -1 → 1 → -1 ...

def main():
    while True:
        opc = exibir_menu()

        if opc == '1':
            jogar_humano_vs_humano()
        elif opc == '2':
            jogar_contra_maquina()
        elif opc == '3':
            print("\nEscolha a dificuldade do Minimax:")
            print("1 - Médio (50% minimax, 50% aleatório)")
            print("2 - Difícil (sempre minimax)")
            escolha = input("Escolha (1/2): ")
            modo = {'1':'medio', '2':'dificil'}.get(escolha, 'dificil')
            jogar_contra_minimax(modo)
        elif opc == '4':
            print("\n Rede Neural ainda não implementada.")
            input("Pressione ENTER.")
        elif opc == '0':
            break
        else:
            print("Opção inválida!")
            input("Pressione ENTER.")

if __name__ == '__main__':
    main()
