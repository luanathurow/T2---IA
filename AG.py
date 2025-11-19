# operacao/genetic_algorithm.py
import random
import numpy as np
import multiprocessing

from tic_tac_toe import TicTacToe
from rede_neural import NeuralNetwork
from operacao.aptidao import aptidao

# ----------------------------------------------------------
# Parâmetros do AG (pode ajustar depois para experimentar)
# ----------------------------------------------------------
POP_SIZE = 40
NUM_GERACOES = 50
ELITE_SIZE = 4
TORNEIO_K = 3
TAXA_MUTACAO = 0.1
STD_MUTACAO = 0.1
JOGOS_POR_INDIVIDUO = 5  # nº de partidas rede vs oponente para avaliar fitness


# ----------------------------------------------------------
# Oponente ALEATÓRIO (sem minimax)
# ----------------------------------------------------------
def jogada_oponente_aleatorio(jogo: TicTacToe, jogador: int):
    movs = jogo.movimentos_disponiveis()
    if not movs:
        return
    l, c = random.choice(movs)
    jogo.fazer_jogada(l, c, jogador)


# A avaliação agora é feita por operacao.aptidao. A função `aptidao`
# avalia um cromossomo jogando contra o Minimax e já retorna um escore.


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
def _eval_crom_tuple(args):
    chrom, partidas, modo, hidden = args
    return aptidao(chrom, partidas=partidas, modo=modo, hidden_size=hidden)


def treinar_ag(
    num_geracoes=NUM_GERACOES,
    pop_size=POP_SIZE,
    hidden_size=18,
    modo_minimax: str = 'dificil',
    partidas_por_individuo: int = JOGOS_POR_INDIVIDUO,
    elite_size: int = ELITE_SIZE,
    torneio_k: int = TORNEIO_K,
    taxa_mutacao: float = TAXA_MUTACAO,
    std_mutacao: float = STD_MUTACAO,
    use_parallel: bool = False,
    num_workers: int = None
):
    """
    Treina a rede usando AG jogando contra um oponente aleatório.

    Retorna:
    - melhor_cromossomo_global (np.ndarray)
    - melhor_rede (instância de NeuralNetwork)
    """
    populacao = inicializar_populacao(
        pop_size=pop_size,
        hidden_size=hidden_size
    )

    melhor_fitness_global = float('-inf')
    melhor_cromossomo_global = None

    for geracao in range(num_geracoes):
        # Avaliação de todos os cromossomos usando a função de aptidão
        if use_parallel:
            # prepara argumentos para cada cromossomo
            args_list = [(chrom, partidas_por_individuo, modo_minimax, hidden_size) for chrom in populacao]
            workers = num_workers or multiprocessing.cpu_count()
            with multiprocessing.Pool(processes=workers) as pool:
                fitness = pool.map(_eval_crom_tuple, args_list)
            fitness = np.array(fitness, dtype=float)
        else:
            fitness = []
            for chrom in populacao:
                fit = aptidao(chrom, partidas=partidas_por_individuo, modo=modo_minimax, hidden_size=hidden_size)
                fitness.append(fit)
            fitness = np.array(fitness, dtype=float)

        fitness = np.array(fitness, dtype=float)
        media_fit = fitness.mean()
        max_fit = fitness.max()
        min_fit = fitness.min()

        print(f"Geração {geracao + 1}/{num_geracoes} | "
              f"fitness: max={max_fit:.2f}, med={media_fit:.2f}, min={min_fit:.2f}")

        # Atualiza melhor global
        idx_best = int(np.argmax(fitness))
        if fitness[idx_best] > melhor_fitness_global:
            melhor_fitness_global = fitness[idx_best]
            melhor_cromossomo_global = populacao[idx_best].copy()

        # Critério simples de convergência (opcional)
        if geracao > 5 and np.std(fitness) < 1e-3:
            print("Convergência detectada (baixa variância de fitness). Encerrando treino.")
            break

        # Gera nova população (com parâmetros customizáveis)
        elite = selecao_elitismo(populacao, fitness, elite_size=elite_size)
        pais = selecao_torneio(populacao, fitness, k=torneio_k)
        filhos = gerar_filhos(pais)
        mutacao(filhos, taxa_mutacao=taxa_mutacao, std_mutacao=std_mutacao)

        populacao = elite + filhos
        populacao = populacao[:pop_size]

    # Constrói rede com o melhor cromossomo encontrado
    melhor_rede = NeuralNetwork.from_chromosome(melhor_cromossomo_global, hidden_size=hidden_size)

    # Salva cromossomo em arquivo para uso pelo front-end
    np.save("best_chromosome.npy", melhor_cromossomo_global)
    print(f"\nTreino concluído. Melhor fitness global = {melhor_fitness_global:.2f}")
    print("Melhor cromossomo salvo em 'best_chromosome.npy'.")

    return melhor_cromossomo_global, melhor_rede
