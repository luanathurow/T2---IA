import numpy as np

class Individuo:
    def __init__(self, num_pesos, random_init=True):
        self.num_pesos = num_pesos
        if random_init:
            self.pesos = np.random.uniform(-1, 1, num_pesos)
        else:
            self.pesos = np.zeros(num_pesos)

        self.aptidao = None

    def calcular_aptidao(self, funcao_aptidao):
        """
        funcao_aptidao será implementada depois
        usando Minimax + rede neural.
        """
        self.aptidao = funcao_aptidao(self.pesos)

    def mutacao(self, taxa_mutacao=0.1):
        for i in range(self.num_pesos):
            if np.random.rand() < taxa_mutacao:
                self.pesos[i] += np.random.normal(0, 0.1)
