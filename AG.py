# operacao/genetic_algorithm.py
import random
import numpy as np

from tic_tac_toe import TicTacToe
from rede_neural import NeuralNetwork
from minimax import melhor_jogada_modo  # <-- novo


# ----------------------------------------------------------
# Parâmetros do AG (pode ajustar depois para experimentar)
# ----------------------------------------------------------
venceu = 0
perdeu = 0
empate = 0
arq = ''
POP_SIZE = 40
NUM_GERACOES = 50
ELITE_SIZE = 4
TORNEIO_K = 3
TAXA_MUTACAO = 0.1
STD_MUTACAO = 0.1
JOGOS_POR_INDIVIDUO = 5  # nº de partidas rede vs oponente para avaliar fitness
log_file = open("log_ag.txt", "w")

def log(msg):
    global log_file
    print(msg)            # continua aparecendo no console
    log_file.write(msg + "\n")

# ----------------------------------------------------------
# Oponente ALEATÓRIO (sem minimax)
# ----------------------------------------------------------
def jogada_oponente_aleatorio(jogo: TicTacToe, jogador: int):
    movs = jogo.movimentos_disponiveis()
    if not movs:
        return
    l, c = random.choice(movs)
    jogo.fazer_jogada(l, c, jogador)

def jogada_oponente_minimax(jogo: TicTacToe, jogador: int, dificuldade: str = 'dificil'):
    """
    Oponente jogando com Minimax (usando sua função melhor_jogada_modo).
    """
    move = melhor_jogada_modo(jogo, jogador, modo=dificuldade)
    if move is None:
        # fallback de segurança
        movs = jogo.movimentos_disponiveis()
        if not movs:
            return
        l, c = random.choice(movs)
    else:
        l, c = move

    jogo.fazer_jogada(l, c, jogador)

# ----------------------------------------------------------
# Função de avaliação (fitness) - SEM MINIMAX
# ----------------------------------------------------------
def avaliar_rede(rede: NeuralNetwork, dificuldade_oponente: str = 'dificil') -> float:
    """
    Avalia a rede jogando JOGOS_POR_INDIVIDUO partidas contra um oponente Minimax.

    - A rede SEMPRE começa e joga como X (1).
    - O oponente (Minimax) joga como O (-1).
    - Retorna um escore de fitness (quanto maior, melhor).
    """
    global venceu, perdeu, empate
    fitness_total = 0.0

    for _ in range(JOGOS_POR_INDIVIDUO):
        jogo = TicTacToe()
        jogador_atual = 1  # rede começa como X

        penalidade_invalidos = 0
        jogadas_validas = 0

        while not jogo.jogo_terminou():
            if jogador_atual == 1:
                # Rede Neural
                movs_validos = jogo.movimentos_disponiveis()
                if not movs_validos:
                    break

                l, c = rede.escolher_jogada(jogo.board, movs_validos)

                if not jogo.jogada_valida(l, c):
                    penalidade_invalidos += 1
                    l, c = random.choice(movs_validos)

                jogo.fazer_jogada(l, c, jogador_atual)
                jogadas_validas += 1

            else:
                # Oponente Minimax
                jogada_oponente_minimax(jogo, jogador_atual, dificuldade_oponente)

            jogador_atual *= -1  # troca turno

        vencedor = jogo.checar_vencedor()

        # ------------------------------
        # Função de aptidão (fitness)
        # ------------------------------
        if vencedor == 1:      # rede venceu
            venceu += 1
            fitness_total += 10
        elif vencedor == -1:   # rede perdeu
            perdeu += 1
            fitness_total -= 8
        else:                  # empate
            empate += 1
            fitness_total += 5

        # penaliza jogadas inválidas e recompensa jogadas válidas
        fitness_total -= 3 * penalidade_invalidos
        fitness_total += 1 * jogadas_validas

    return fitness_total


# ----------------------------------------------------------
# Operadores do AG
# ----------------------------------------------------------
def inicializar_populacao(pop_size=POP_SIZE, input_size=9, hidden_size=18, output_size=9):
    """
    Gera população inicial de cromossomos (vetores 1D de pesos da rede).
    """
    dummy_net = NeuralNetwork(input_size, hidden_size, output_size)
    num_w = dummy_net.num_weights

    populacao = []
    for _ in range(pop_size):
        chrom = np.random.uniform(-1, 1, num_w)  # double em [-1,1]
        populacao.append(chrom)
    return populacao


def selecao_elitismo(populacao, fitness, elite_size=ELITE_SIZE):
    """
    Seleciona os 'elite_size' melhores indivíduos (cromossomos).
    """
    fitness = np.asarray(fitness)
    idx_ordenados = np.argsort(fitness)[::-1]  # maior -> menor
    elite = [populacao[i].copy() for i in idx_ordenados[:elite_size]]
    return elite


def selecao_torneio(populacao, fitness, k=TORNEIO_K):
    """
    Seleção por torneio de tamanho k.
    """
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
    """
    Cruzamento aritmético simples:
    filho1 = alpha * pai1 + (1 - alpha) * pai2
    filho2 = alpha * pai2 + (1 - alpha) * pai1
    """
    alpha = 0.5
    filho1 = alpha * pai1 + (1 - alpha) * pai2
    filho2 = alpha * pai2 + (1 - alpha) * pai1
    return filho1, filho2


def gerar_filhos(pais):
    """
    Gera nova população de filhos a partir da lista de pais.
    """
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
            # Se sobrar um, apenas copia (sem cruzar)
            filhos.append(pais[i].copy())
    return filhos


def mutacao(populacao, taxa_mutacao=TAXA_MUTACAO, std_mutacao=STD_MUTACAO):
    """
    Mutação gaussiana: adiciona ruído N(0, std_mutacao) em uma fração dos genes.
    """
    for chrom in populacao:
        mask = np.random.rand(chrom.size) < taxa_mutacao
        ruido = np.random.normal(0, std_mutacao, chrom.size)
        chrom += mask * ruido


# ----------------------------------------------------------
# Treino principal do AG (SEM MINIMAX)
# ----------------------------------------------------------
def treinar_ag(
    num_geracoes=NUM_GERACOES,
    pop_size=POP_SIZE,
    hidden_size=18
):
    global venceu, perdeu, empate
    """
    Treina a rede usando AG jogando contra um oponente Minimax.

    - Nas primeiras gerações → Minimax modo 'medio'
    - Nas últimas gerações   → Minimax modo 'dificil'

    Retorna:
    - melhor_cromossomo_global (np.ndarray)
    - melhor_rede (instância de NeuralNetwork)
    """
    populacao = inicializar_populacao(
        pop_size=pop_size,
        hidden_size=hidden_size
    )
    historico_diferencas = []   # armazena as diferenças entre populações
    populacao_anterior = None   # primeira geração não tem anterior

    melhor_fitness_global = float('-inf')
    melhor_cromossomo_global = None
    
    # ponto de troca: metade das gerações
    ponto_troca = max(1, num_geracoes // 2)

    for geracao in range(num_geracoes):
        fitness = []

        # Define dificuldade do oponente nesta geração
        if geracao < ponto_troca:
            dificuldade_atual = 'medio'
        else:
            dificuldade_atual = 'dificil'

        print(f"\n>>> Geração {geracao+1}/{num_geracoes} "
              f"jogando contra Minimax modo '{dificuldade_atual}'")

        # ---------------------------------------------
        # MÉDIA DA DIFERENÇA ENTRE GERAÇÕES
        # ---------------------------------------------
        if populacao_anterior is not None:
            pa = np.array(populacao_anterior)
            pc = np.array(populacao)

            diffs = np.linalg.norm(pc - pa, axis=1)
            media_dif = diffs.mean()
            historico_diferencas.append(media_dif)

            print(f"MÉDIA DA DIFERENÇA PARA A GERAÇÃO {geracao+1}: {media_dif:.4f}")

        populacao_anterior = [chrom.copy() for chrom in populacao]

        # Avaliação de todos os cromossomos
        for chrom in populacao:
            rede = NeuralNetwork.from_chromosome(chrom, hidden_size=hidden_size)
            fit = avaliar_rede(rede, dificuldade_oponente=dificuldade_atual)
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
        venceu = 0
        perdeu = 0
        empate = 0

        # Atualiza melhor global
        idx_best = int(np.argmax(fitness))
        if fitness[idx_best] > melhor_fitness_global:
            melhor_fitness_global = fitness[idx_best]
            melhor_cromossomo_global = populacao[idx_best].copy()

        # Critério simples de convergência (opcional)
        if geracao > 5 and np.std(fitness) < 1e-3:
            print("Convergência detectada (baixa variância de fitness). Encerrando treino.")
            break

        # Gera nova população
        elite = selecao_elitismo(populacao, fitness)
        pais = selecao_torneio(populacao, fitness)
        filhos = gerar_filhos(pais)
        # mutacao(filhos)
        populacao = elite + filhos
        populacao = populacao[:pop_size]

    melhor_rede = NeuralNetwork.from_chromosome(melhor_cromossomo_global, hidden_size=hidden_size)

    np.save("best_chromosome.npy", melhor_cromossomo_global)
    print(f"\nTreino concluído. Melhor fitness global = {melhor_fitness_global:.2f}")
    print("Melhor cromossomo salvo em 'best_chromosome.npy'.")

    return melhor_cromossomo_global, melhor_rede
