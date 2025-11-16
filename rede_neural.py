import numpy as np

class RedeNeural:
    def __init__(self, pesos):
        """
        pesos = vetor de 180 floats (cromossomo do AG)
        """
        self.pesos = pesos
        self._decodificar_pesos()

    def _decodificar_pesos(self):
        """
        Divide o vetor de 180 pesos em:
        - W1 (9x9)
        - b1 (9)
        - W2 (9x9)
        - b2 (9)
        """
        idx = 0

        # Entrada -> Oculta (81 pesos)
        self.W1 = self.pesos[idx:idx+81].reshape(9, 9)
        idx += 81

        # Bias da oculta (9)
        self.b1 = self.pesos[idx:idx+9]
        idx += 9

        # Oculta -> Saída (81 pesos)
        self.W2 = self.pesos[idx:idx+81].reshape(9, 9)
        idx += 81

        # Bias da saída (9)
        self.b2 = self.pesos[idx:idx+9]

    # Funções de ativação
    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, tabuleiro_flat):
        # tabuleiro_flat = vetor com 9 valores (X=1, O=-1, vazio=0)

        # Camada oculta
        hidden = np.dot(tabuleiro_flat, self.W1) + self.b1
        hidden = self.relu(hidden)

        # Camada de saída
        out = np.dot(hidden, self.W2) + self.b2

        return out  # 9 scores (um por casa)

    def escolher_jogada(self, tabuleiro_flat, movimentos_validos):
        saida = self.forward(tabuleiro_flat)

        # Filtra só posições válidas
        melhores_indices = sorted(
            movimentos_validos,
            key=lambda pos: saida[pos[0]*3 + pos[1]],
            reverse=True
        )

        # Retorna a melhor posição
        return melhores_indices[0]
