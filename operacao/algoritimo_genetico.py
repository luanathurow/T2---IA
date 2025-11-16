import random
import numpy as np

from dominio.individuo import Individuo
from dominio.populacao import Populacao

class AlgoritmoGenetico:
    def __init__(self, tamanho_pop, num_pesos, taxa_cross=0.7, taxa_mut=0.1, elitismo=True):
        self.tamanho_pop = tamanho_pop
        self.num_pesos = num_pesos
        self.taxa_cross = taxa_cross
        self.taxa_mut = taxa_mut
        self.elitismo = elitismo

    # Seleção por torneio 
    def selecao_torneio(self, populacao, k=2):
        s = random.sample(populacao.individuos, k)
        s.sort(key=lambda ind: ind.aptidao, reverse=True)
        return s[0]

    # Crossover de ponto de corte
    def crossover(self, pai1, pai2):
        ponto = random.randint(0, self.num_pesos-1)

        filho1 = Individuo(self.num_pesos, random_init=False)
        filho2 = Individuo(self.num_pesos, random_init=False)

        filho1.pesos[:ponto] = pai1.pesos[:ponto]
        filho1.pesos[ponto:] = pai2.pesos[ponto:]

        filho2.pesos[:ponto] = pai2.pesos[:ponto]
        filho2.pesos[ponto:] = pai1.pesos[ponto:]

        return filho1, filho2

    def nova_geracao(self, populacao, funcao_aptidao):
        
        nova_pop = Populacao(0, self.num_pesos, individuos_aleatorios=False)

        if self.elitismo:
            nova_pop.individuos.append(populacao.melhor())

        while len(nova_pop.individuos) < self.tamanho_pop:
            pai1 = self.selecao_torneio(populacao)
            pai2 = self.selecao_torneio(populacao)

            if random.random() < self.taxa_cross:
                filho1, filho2 = self.crossover(pai1, pai2)
            else:
                filho1 = pai1
                filho2 = pai2

            filho1.mutacao(self.taxa_mut)
            filho2.mutacao(self.taxa_mut)

            filho1.calcular_aptidao(funcao_aptidao)
            filho2.calcular_aptidao(funcao_aptidao)

            nova_pop.individuos.append(filho1)
            if len(nova_pop.individuos) < self.tamanho_pop:
                nova_pop.individuos.append(filho2)

        nova_pop.ordenar()
        return nova_pop

    def executar(self, geracoes, populacao, funcao_aptidao):
        # Avalia aptidão inicial
        for ind in populacao.individuos:
            ind.calcular_aptidao(funcao_aptidao)

        populacao.ordenar()

        for g in range(geracoes):
            print(f"\nGeração {g+1}")
            print(f"Melhor aptidão: {populacao.melhor().aptidao:.4f}")

            populacao = self.nova_geracao(populacao, funcao_aptidao)

        return populacao
