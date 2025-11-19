# main.py

from tic_tac_toe import TicTacToe
from minimax import melhor_jogada_modo
from AG import treinar_ag
from rede_neural import NeuralNetwork

import numpy as np
import random
import os

# Mensagem de pausa reutilizável
PAUSE_MSG = "Pressione ENTER para continuar."


# ----------------------------------------------------------------------
# Utilitários de console
# ----------------------------------------------------------------------
def limpar_console():
    os.system('cls' if os.name == 'nt' else 'clear')


def exibir_menu():
    print("\n===== JOGO DA VELHA - IA =====")
    print("1 - Jogar contra outro humano")
    print("2 - Jogar contra máquina (aleatória)")
    print("3 - Jogar contra IA (Minimax)  [Dificuldade 1]")
    print("4 - Jogar contra IA (Rede Neural treinada)  [Dificuldade 2]")
    print("5 - Treinar Rede Neural com Algoritmo Genético + Minimax")
    print("0 - Sair")
    return input("Escolha uma opção: ")


# ----------------------------------------------------------------------
# Jogada da máquina ALEATÓRIA
# ----------------------------------------------------------------------
def jogada_maquina_aleatoria(jogo: TicTacToe, jogador):
    movimentos = jogo.movimentos_disponiveis()
    if not movimentos:
        return
    l, c = random.choice(movimentos)
    jogo.fazer_jogada(l, c, jogador)


# ----------------------------------------------------------------------
# Jogada do humano
# ----------------------------------------------------------------------
def jogada_humano(jogo: TicTacToe, jogador):
    while True:
        try:
            movs = jogo.movimentos_disponiveis()
            print(f"Movimentos válidos (0-based): {movs}")
            pos = input("Digite linha e coluna 0-based (ex: 0 2): ")
            l, c = map(int, pos.split())

            # aceitar apenas 0-based (0..2)
            if l in range(3) and c in range(3):
                if jogo.fazer_jogada(l, c, jogador):
                    return
                else:
                    print("Casa ocupada! Tente de novo.")
            else:
                print("Posição inválida! Use valores entre 0 e 2.")
        except Exception:
            print("Entrada inválida! Use o formato: linha coluna (ex: 0 2)")


# ----------------------------------------------------------------------
# Jogada do Minimax (com modos)
# ----------------------------------------------------------------------
def jogada_minimax(jogo: TicTacToe, jogador, modo: str = 'dificil'):
    move = melhor_jogada_modo(jogo, jogador, modo)
    if move is None:
        # fallback para aleatória (não deve ocorrer se movimentos_disponiveis estiver correto)
        jogada_maquina_aleatoria(jogo, jogador)
    else:
        l, c = move
        jogo.fazer_jogada(l, c, jogador)


# ----------------------------------------------------------------------
# Lógica: humano x máquina ALEATÓRIA
# ----------------------------------------------------------------------
def jogar_contra_maquina():
    jogo = TicTacToe()
    jogador = 1  # começa sempre o X (humano)

    while True:
        limpar_console()
        jogo.mostrar()

        if jogador == 1:
            print("Sua vez (X)")
            jogada_humano(jogo, 1)
        else:
            print("Vez da máquina (O) - aleatória")
            jogada_maquina_aleatoria(jogo, -1)

        vencedor = jogo.checar_vencedor()
        if vencedor is not None:
            limpar_console()
            jogo.mostrar()
            if vencedor == 1:
                print("\nVocê venceu! 🎉")
            else:
                print("\nA máquina venceu! 🤖")
            input(PAUSE_MSG)
            return vencedor

        if jogo.checar_empate():
            limpar_console()
            jogo.mostrar()
            print("\nEmpate!")
            input(PAUSE_MSG)
            return 0

        jogador *= -1  # troca 1 → -1 → 1 → -1 ...


# ----------------------------------------------------------------------
# Lógica: humano x humano
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
            input(PAUSE_MSG)
            return vencedor

        if jogo.checar_empate():
            limpar_console()
            jogo.mostrar()
            print("\nEmpate!")
            input(PAUSE_MSG)
            return 0

        jogador *= -1


# ----------------------------------------------------------------------
# Lógica: humano x IA (Minimax)  → Dificuldade 1
# ----------------------------------------------------------------------
def jogar_contra_minimax():
    print("\nEscolha o modo do Minimax:")
    print("1 - Médio (50% minimax, 50% aleatório)")
    print("2 - Difícil (sempre minimax)")
    escolha = input("Escolha (1/2): ")
    modo = {'1': 'medio', '2': 'dificil'}.get(escolha, 'dificil')

    jogo = TicTacToe()
    jogador = 1  # começa sempre o X (humano)

    while True:
        limpar_console()
        jogo.mostrar()

        if jogador == 1:
            print("Sua vez (X)")
            jogada_humano(jogo, 1)
        else:
            print(f"Vez da IA Minimax ({modo}) (O)")
            jogada_minimax(jogo, -1, modo)

        vencedor = jogo.checar_vencedor()
        if vencedor is not None:
            limpar_console()
            jogo.mostrar()
            if vencedor == 1:
                print("\nVocê venceu! 🎉")
            else:
                print("\nA IA Minimax venceu! 🤖")
            input(PAUSE_MSG)
            return vencedor

        if jogo.checar_empate():
            limpar_console()
            jogo.mostrar()
            print("\nEmpate!")
            input(PAUSE_MSG)
            return 0

        jogador *= -1  # troca 1 → -1 → 1 → -1 ...


# ----------------------------------------------------------------------
# Rede Neural: carregar melhor cromossomo salvo pelo AG
# ----------------------------------------------------------------------
def carregar_melhor_rede(hidden_size=18):
    try:
        chrom = np.load("best_chromosome.npy")
        rede = NeuralNetwork.from_chromosome(chrom, hidden_size=hidden_size)
        return rede
    except Exception as e:
        print("\n[ERRO] Não foi possível carregar best_chromosome.npy")
        print("Detalhes:", e)
        input("Pressione ENTER para continuar.")
        return None


# ----------------------------------------------------------------------
# Lógica: humano x IA (Rede Neural)  → Dificuldade 2
# ----------------------------------------------------------------------
def jogar_contra_rede():
    rede = carregar_melhor_rede()
    if rede is None:
        return

    jogo = TicTacToe()
    # Para destacar a “dificuldade 2”, deixamos a REDE começar como X
    jogador = 1  # rede (X) começa
    humano = -1  # humano joga com O

    jogadas_totais_rede = 0
    jogadas_validas_rede = 0

    while True:
        limpar_console()
        jogo.mostrar()

        if jogador == 1:
            print("Vez da IA Rede Neural (X)")
            movs_validos = jogo.movimentos_disponiveis()
            if not movs_validos:
                break
            l, c = rede.escolher_jogada(jogo.board, movs_validos)

            if not jogo.jogada_valida(l, c):
                l, c = random.choice(movs_validos)
            jogo.fazer_jogada(l, c, 1)

            jogadas_totais_rede += 1
            jogadas_validas_rede += 1
        else:
            print("Sua vez (O)")
            jogada_humano(jogo, humano)

        vencedor = jogo.checar_vencedor()
        if vencedor is not None:
            limpar_console()
            jogo.mostrar()
            if vencedor == 1:
                print("\nA IA Rede Neural venceu! 🤖🧠")
            else:
                print("\nVocê venceu! 🎉")
            input(PAUSE_MSG)
            return vencedor

        if jogo.checar_empate():
            limpar_console()
            jogo.mostrar()
            print("\nEmpate!")
            input(PAUSE_MSG)
            return 0

        jogador *= -1

    # Métrica simples de “acurácia”: proporção de jogadas válidas da rede
    if jogadas_totais_rede > 0:
        acuracia = jogadas_validas_rede / jogadas_totais_rede
        print(f"\nAcurácia aproximada da IA (jogadas válidas): {acuracia * 100:.2f}%")
    else:
        print("\nNão houve jogadas da IA para medir acurácia.")

    input("Pressione ENTER para continuar.")


# ----------------------------------------------------------------------
# Treinar Rede Neural (AG + Minimax)
# ----------------------------------------------------------------------
def treinar_rede():
    limpar_console()
    print("=== Treino da Rede Neural com Algoritmo Genético + Minimax ===\n")
    print("Durante o treino, a rede SEMPRE começa jogando como X (1).")
    print("O adversário usado pelo AG é o Minimax (primeiro médio, depois difícil).")
    print("Os pesos finais serão salvos em 'best_chromosome.npy'.\n")
    input("Pressione ENTER para iniciar o treino...")

    melhor_chrom, melhor_rede = treinar_ag()

    print("\nTreino concluído!")
    print("Melhor cromossomo salvo em 'best_chromosome.npy'.")
    input("Pressione ENTER para continuar.")


# ----------------------------------------------------------------------
# Main / menu principal
# ----------------------------------------------------------------------
def main():
    while True:
        limpar_console()
        opc = exibir_menu()

        if opc == '1':
            _res = jogar_humano_vs_humano()
            input("Pressione ENTER para continuar.")
        elif opc == '2':
            _res = jogar_contra_maquina()
            input("Pressione ENTER para continuar.")
        elif opc == '3':
            # Dificuldade 1: IA Minimax
            _res = jogar_contra_minimax()
            input("Pressione ENTER para continuar.")
        elif opc == '4':
            # Dificuldade 2: IA Rede Neural (treinada pelo AG+Minimax)
            _res = jogar_contra_rede()
            input("Pressione ENTER para continuar.")
        elif opc == '5':
            treinar_rede()
        elif opc == '0':
            break
        else:
            print("Opção inválida!")
            input("Pressione ENTER.")


if __name__ == '__main__':
    main()
