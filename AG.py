# operacao/genetic_algorithm.py
import random
import numpy as np

from tic_tac_toe import TicTacToe
from rede_neural import NeuralNetwork
from minimax import melhor_jogada_modo


# ----------------------------------------------------------
# Parâmetros do AG
# ----------------------------------------------------------
venceu = 0
perdeu = 0
empate = 0

POP_SIZE = 12
NUM_GERACOES = 20
ELITE_SIZE = 4
TORNEIO_K = 3
TAXA_MUTACAO = 0.1
STD_MUTACAO = 0.1
JOGOS_POR_INDIVIDUO = 40  # nº de partidas rede vs oponente para avaliar fitness


def log(msg):
    print(msg)


# ----------------------------------------------------------
# Função de aptidão (fitness)
# ----------------------------------------------------------
def avaliar_rede(
    rede: NeuralNetwork,
    jogos_por_individuo: int = JOGOS_POR_INDIVIDUO
) -> float:
    """
    Função de aptidão (fitness) de um cromossomo (rede neural).

    A rede joga 'jogos_por_individuo' partidas contra o oponente:

      - Oponente: Minimax em modo 'medio' (50% Minimax, 50% aleatório)
      - Em cada partida, sorteia se a rede joga como X (1) ou O (-1).
      - X (1) SEMPRE começa a partida.

    Métrica:
      • Vitória da rede  : +1.0
      • Empate           : +0.5
      • Derrota          : -1.0
      • Jogada inválida  : -10.0 por tentativa (escolher célula ocupada)
    """
    global venceu, perdeu, empate

    fitness_total = 0.0

    for _ in range(jogos_por_individuo):
        jogo = TicTacToe()

        rede_jogador = random.choice([1, -1])  # 1 = X, -1 = O
        oponente = -rede_jogador

        jogador_atual = 1  # X sempre começa
        penalidade_invalidos = 0

        while not jogo.jogo_terminou():
            movs_validos = jogo.movimentos_disponiveis()
            if not movs_validos:
                break

            if jogador_atual == rede_jogador:
                # ---------- Vez da Rede Neural ----------
                l, c = rede.escolher_jogada(jogo.board, movs_validos)

                # Se escolher célula inválida (ocupada ou fora da lista), penaliza
                if not jogo.jogada_valida(l, c):
                    penalidade_invalidos += 1
                    l, c = random.choice(movs_validos)

                jogo.fazer_jogada(l, c, rede_jogador)

            else:
                # --- Vez do OPONENTE (50% Minimax, 50% aleatório) ---
                move = melhor_jogada_modo(jogo, jogador_atual, modo='medio')
                if move is None:
                    l, c = random.choice(movs_validos)
                else:
                    l, c = move

                jogo.fazer_jogada(l, c, jogador_atual)

            jogador_atual *= -1

        vencedor = jogo.checar_vencedor()

        # --------- Pontuação por resultado do jogo ---------
        if vencedor == rede_jogador:      # rede venceu
            venceu += 1
            fitness_total += 1.0
        elif vencedor == oponente:        # rede perdeu
            perdeu += 1
            fitness_total -= 1.0
        else:                             # empate
            empate += 1
            fitness_total += 0.5

        # --------- Penalização por jogadas inválidas ---------
        fitness_total -= 10.0 * penalidade_invalidos

    return fitness_total


# ----------------------------------------------------------
# Operadores do AG
# ----------------------------------------------------------
def inicializar_populacao(pop_size=POP_SIZE, input_size=9, hidden_size=18, output_size=9):
    dummy_net = NeuralNetwork(input_size, hidden_size, output_size)
    num_w = dummy_net.num_weights

    populacao = []
    for _ in range(pop_size):
        chrom = np.random.uniform(-1, 1, num_w)
        populacao.append(chrom)
    return populacao


def selecao_elitismo(populacao, fitness, elite_size=ELITE_SIZE):
    fitness = np.asarray(fitness)
    idx_ordenados = np.argsort(fitness)[::-1]  # maior -> menor
    elite = [populacao[i].copy() for i in idx_ordenados[:elite_size]]
    return elite


def selecao_torneio(populacao, fitness, k=TORNEIO_K):
    fitness = np.asarray(fitness)
    selecionados = []
    pop_size = len(populacao)

    for _ in range(pop_size):
        competidores_idx = np.random.randint(0, pop_size, k)
        melhor = competidores_idx[0]
        for idx in competidores_idx[1:]:
            if fitness[idx] > fitness[melhor]:
                melhor = idx
        selecionados.append(populacao[melhor].copy())

    return selecionados


def cruzamento_aritmetico(pai1, pai2):
    alpha = 0.5
    filho1 = alpha * pai1 + (1 - alpha) * pai2
    filho2 = alpha * pai2 + (1 - alpha) * pai1
    return filho1, filho2


def gerar_filhos(pais):
    random.shuffle(pais)
    filhos = []
    for i in range(0, len(pais), 2):
        if i + 1 < len(pais):
            p1 = pais[i]
            p2 = pais[i + 1]
            f1, f2 = cruzamento_aritmetico(p1, p2)
            filhos.append(f1)
            filhos.append(f2)
        else:
            filhos.append(pais[i].copy())
    return filhos


def mutacao(populacao, taxa_mutacao=TAXA_MUTACAO, std_mutacao=STD_MUTACAO):
    for chrom in populacao:
        mask = np.random.rand(chrom.size) < taxa_mutacao
        ruido = np.random.normal(0, std_mutacao, chrom.size)
        chrom += mask * ruido


# ----------------------------------------------------------
# Treino principal do AG contra Minimax (50% aleatório)
# ----------------------------------------------------------
def treinar_ag(
    num_geracoes=NUM_GERACOES,
    pop_size=POP_SIZE,
    hidden_size=18
):
    global venceu, perdeu, empate

    populacao = inicializar_populacao(
        pop_size=pop_size,
        hidden_size=hidden_size
    )

    populacao_anterior = None
    historico_diferencas = []

    melhor_fitness_global = float('-inf')
    melhor_cromossomo_global = None

    for geracao in range(num_geracoes):
        fitness = []

        print(
            f">>> Geração {geracao+1}/{num_geracoes} "
            f"(Oponente: Minimax modo 'medio' – 50% Minimax / 50% aleatório)"
        )

        # 1) Avaliar TODOS os indivíduos da geração atual
        for chrom in populacao:
            rede = NeuralNetwork.from_chromosome(chrom, hidden_size=hidden_size)
            fit = avaliar_rede(rede)
            fitness.append(fit)

        fitness = np.array(fitness, dtype=float)
        media_fit = fitness.mean()
        max_fit = fitness.max()
        min_fit = fitness.min()

        idx_ordenados = np.argsort(fitness)[::-1]

        print(f"\n=== Geração {geracao+1}/{num_geracoes} ===")
        print(f"Melhor fitness : {max_fit:.3f}")
        print(f"Média          : {media_fit:.3f}")
        print(f"Pior fitness   : {min_fit:.3f}\n")

        print("ELITE (top indivíduos):")
        for rank in range(ELITE_SIZE):
            idx = idx_ordenados[rank]
            print(f"  #{rank+1:02d} Indiv {idx:02d} → fitness = {fitness[idx]:.3f}")

        print("\nDEMAIS INDIVÍDUOS:")
        for rank in range(ELITE_SIZE, len(populacao)):
            idx = idx_ordenados[rank]
            print(f"  Indiv {idx:02d} → fitness = {fitness[idx]:.3f}")

        print(f'Venceu {venceu}  Perdeu {perdeu} Empatou {empate}')
        venceu = perdeu = empate = 0

        # 2) Diferença média entre gerações
        if populacao_anterior is not None:
            pa = np.array(populacao_anterior)
            pc = np.array(populacao)
            diffs = np.linalg.norm(pc - pa, axis=1)
            media_dif = diffs.mean()
            historico_diferencas.append(media_dif)
            print(
                f"MÉDIA DA DIFERENÇA ENTRE GERAÇÃO {geracao} "
                f"e {geracao+1}: {media_dif:.4f}"
            )

        # 3) Atualiza melhor global
        idx_best = int(np.argmax(fitness))
        if fitness[idx_best] > melhor_fitness_global:
            melhor_fitness_global = fitness[idx_best]
            melhor_cromossomo_global = populacao[idx_best].copy()

        if geracao > 10 and np.std(fitness) < 1e-2:
            print("Convergência detectada (baixa variância de fitness). Encerrando treino.")
            break

        # 4) Nova geração
        elite = selecao_elitismo(populacao, fitness)
        pais = selecao_torneio(populacao, fitness)
        filhos = gerar_filhos(pais)
        mutacao(filhos)

        populacao_anterior = [chrom.copy() for chrom in populacao]

        populacao = elite + filhos
        populacao = populacao[:pop_size]

    melhor_rede = NeuralNetwork.from_chromosome(melhor_cromossomo_global, hidden_size=hidden_size)

    np.save("best_chromosome.npy", melhor_cromossomo_global)
    print(f"\nTreino concluído. Melhor fitness global = {melhor_fitness_global:.2f}")
    print("Melhor cromossomo salvo em 'best_chromosome.npy'.")

    return melhor_cromossomo_global, melhor_rede
