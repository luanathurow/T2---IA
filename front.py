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
def minimax_vs_rn(jogo: TicTacToe, jogador, pesos=None, modo: str = 'dificil'):
    """Executa o restante da partida fazendo Minimax jogar contra uma Rede Neural.

    - `jogo`: instância de `TicTacToe` com estado atual
    - `jogador`: inteiro (1 ou -1) que indica qual lado será jogado pelo Minimax
    - `pesos`: sequência de 180 floats ou `None`. Se `None`, tentamos carregar
       `pesos.npy` do diretório do projeto; se não disponível, usamos pesos aleatórios.
    - `modo`: modo do Minimax passado para `melhor_jogada_modo` ('dificil'|'medio')

    A função joga até o final, alternando Minimax e Rede Neural, e aplica as jogadas
    diretamente em `jogo`.
    """

    # Carrega Rede Neural
    from rede_neural import RedeNeural


    # Tenta obter pesos: argumento -> arquivo `pesos.npy` -> aleatórios
    rn_pesos = None
    if pesos is not None:
        rn_pesos = pesos
    else:
        try:
            import numpy as _np
            import pathlib
            p = pathlib.Path(__file__).resolve().parent.parent / 'pesos.npy'
            if p.exists():
                rn_pesos = _np.load(str(p))
        except Exception:
            rn_pesos = None

    if rn_pesos is None:
        # fallback: pesos aleatórios de sinal e magnitude variados
        try:
            import numpy as _np
            rn_pesos = _np.random.uniform(-1, 1, 180)
        except Exception:
            # sem numpy disponível, usa lista de floats via random
            rn_pesos = [random.uniform(-1, 1) for _ in range(180)]

    rn = RedeNeural(rn_pesos)

    # Importa Minimax
    from operacao.minimax import melhor_jogada_modo

    current = jogador

    # Joga até terminar, alternando Minimax (current==jogador) e RN (current!=jogador)
    while not jogo.jogo_terminou():
        if current == jogador:
            move = melhor_jogada_modo(jogo, current, modo)
            if move is None:
                # sem movimentos (deveria terminar), sai
                break
            jogo.fazer_jogada(move[0], move[1], current)
        else:
            tab_flat = [jogo.board[i][j] for i in range(3) for j in range(3)]
            movs = jogo.movimentos_disponiveis()
            if not movs:
                break
            escolha = rn.escolher_jogada(tab_flat, movs)
            if escolha is None or not jogo.jogada_valida(escolha[0], escolha[1]):
                # fallback aleatório
                l, c = random.choice(movs)
                jogo.fazer_jogada(l, c, current)
            else:
                jogo.fazer_jogada(escolha[0], escolha[1], current)

        current = -current

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
        movs = jogo.movimentos_disponiveis()
        if movs:
            l, c = random.choice(movs)
            jogo.fazer_jogada(l, c, jogador)
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
            # executa Minimax x Rede Neural a partir do estado atual
            minimax_vs_rn(jogo, -1)

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
