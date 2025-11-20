# main.py

from tic_tac_toe import TicTacToe
from minimax import melhor_jogada_modo
from AG import treinar_ag
from rede_neural import NeuralNetwork

import numpy as np
import random
import os


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
            print(f"Movimentos válidos (base 0): {movs}")
            pos = input("Digite linha e coluna base 0 (ex: 0 2): ")
            l, c = map(int, pos.split())

            # aceitar apenas base 0 (0..2)
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
            break

        if jogo.checar_empate():
            limpar_console()
            jogo.mostrar()
            print("\nEmpate!")
            break

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
            break

        if jogo.checar_empate():
            limpar_console()
            jogo.mostrar()
            print("\nEmpate!")
            break

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
            break

        if jogo.checar_empate():
            limpar_console()
            jogo.mostrar()
            print("\nEmpate!")
            break

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

    # Sorteia quem será X e quem será O
    # X (1) sempre começa a partida
    if random.random() < 0.5:
        rede_jogador = 1   # IA é X
        humano = -1        # humano é O
        print("Nesta partida, a IA (Rede Neural) será X e começa jogando.")
    else:
        rede_jogador = -1  # IA é O
        humano = 1         # humano é X
        print("Nesta partida, você será X e começa jogando. A IA é O.")

    input("Pressione ENTER para começar o jogo...")

    jogador = 1  # X sempre começa (pode ser humano ou IA, depende do sorteio)

    while True:
        limpar_console()
        jogo.mostrar()

        if jogador == rede_jogador:
            simbolo = 'X' if rede_jogador == 1 else 'O'
            print(f"Vez da IA Rede Neural ({simbolo})")
            movs_validos = jogo.movimentos_disponiveis()
            if not movs_validos:
                break

            l, c = rede.escolher_jogada(jogo.board, movs_validos)

            # Segurança: se por algum bug vier jogada inválida, corrige
            if not jogo.jogada_valida(l, c):
                l, c = random.choice(movs_validos)

            jogo.fazer_jogada(l, c, rede_jogador)

        else:
            simbolo = 'X' if humano == 1 else 'O'
            print(f"Sua vez ({simbolo})")
            jogada_humano(jogo, humano)

        vencedor = jogo.checar_vencedor()
        if vencedor is not None:
            limpar_console()
            jogo.mostrar()
            if vencedor == rede_jogador:
                print("\nA IA Rede Neural venceu! 🤖🧠")
            else:
                print("\nVocê venceu! 🎉")
            break

        if jogo.checar_empate():
            limpar_console()
            jogo.mostrar()
            print("\nEmpate!")
            break

        jogador *= -1  # alterna turno

    print("\nJogo encerrado.")
    input("Pressione ENTER para continuar.")

# ----------------------------------------------------------------------
# Treinar Rede Neural (AG + Minimax)
# ----------------------------------------------------------------------
def treinar_rede():
    limpar_console()
    print("=== Treino da Rede Neural com Algoritmo Genético + Minimax ===\n")
    print("Durante o treino, em cada partida a rede pode ser X (1) ou O (-1).")
    print("X sempre começa a partida.")
    print("O oponente é o Minimax em modo DIFÍCIL (sempre Minimax).")
    print("Os pesos finais serão salvos em 'best_chromosome.npy'.\n")
    input("Pressione ENTER para iniciar o treino...")

    melhor_chrom, melhor_rede = treinar_ag(hidden_size=18)
    print("\nTreino concluído!")
    print("Melhor cromossomo salvo em 'best_chromosome.npy'.")
    input("Pressione ENTER para continuar.")


def treinar_rede_cust(num_geracoes,pop_size,hidden_size,modo_minimax,partidas_por_individuo,elite_size,torneio_k,taxa_mutacao,std_mutacao,use_parallel,num_workers):
    if use_parallel.lower() == 'true':
        use_parallel = True
    elif use_parallel.lower() == 'false':
        use_parallel = False 

    
    limpar_console()
    print("=== Treino da Rede Neural com Algoritmo Genético + Minimax ===\n")
    print("Durante o treino, em cada partida a rede pode ser X (1) ou O (-1).")
    print("X sempre começa a partida.")
    # Mostra o modo efetivamente escolhido pelo usuário
    mode_desc = "(50% Minimax / 50% aleatório)" if modo_minimax == 'medio' else "(sempre Minimax)"
    print(f"O oponente é o Minimax em modo {modo_minimax.upper()} {mode_desc}.")
    print("Os pesos finais serão salvos em 'best_chromosome.npy'.\n")
    input("Pressione ENTER para iniciar o treino...")

    _, _ = treinar_ag(num_geracoes,pop_size,hidden_size,modo_minimax,partidas_por_individuo,elite_size,torneio_k,taxa_mutacao,std_mutacao,use_parallel,num_workers)
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
            jogar_humano_vs_humano()
        elif opc == '2':
            jogar_contra_maquina()
        elif opc == '3':
            # Dificuldade 1: IA Minimax
            jogar_contra_minimax()
        elif opc == '4':
            # Dificuldade 2: IA Rede Neural (treinada pelo AG+Minimax)
            jogar_contra_rede()
        elif opc == '5':
            opcust = input('Deseja customizar os parametros do treino? s/n').lower()
            if opcust.strip() == 's':
                print("Exemplo de input: num_geracoes pop_size hidden_size modo_minimax(medio ou dificil) partidas_por_individuo elite_size torneio_k taxa_mutacao std_mutacao use_parallel(True ou False) num_workers")
                cust = input('Digite aqui: ').split(' ')
                treinar_rede_cust(int(cust[0]),int(cust[1]),int(cust[2]),cust[3],int(cust[4]),int(cust[5]),int(cust[6]),float(cust[7]),float(cust[8]),cust[9],int(cust[10]))
            else:
                treinar_rede()
            
        elif opc == '0':
            break
        else:
            print("Opção inválida!")
            input("Pressione ENTER.")


if __name__ == '__main__':
    main()
