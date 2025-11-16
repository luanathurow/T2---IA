from dominio.individuo import Individuo


class Populacao:
    def __init__(self, tamanho, num_pesos, individuos_aleatorios=True):
        self.individuos = []
        for _ in range(tamanho):
            ind = Individuo(num_pesos, random_init=individuos_aleatorios)
            self.individuos.append(ind)

    def ordenar(self):
        self.individuos.sort(key=lambda ind: ind.aptidao, reverse=True)

    def melhor(self):
        return self.individuos[0]

    def pior(self):
        return self.individuos[-1]

    def media_aptidao(self):
        return sum(ind.aptidao for ind in self.individuos) / len(self.individuos)
