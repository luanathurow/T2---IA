# operacao/neural_network.py

import numpy as np


class NeuralNetwork:
    """
    MLP de 2 camadas:
    - Entrada: 9 neurônios (tabuleiro 3x3 flatten)
    - Oculta: hidden_size
    - Saída: 9 neurônios (uma saída por casa do tabuleiro)

    NÃO usa backpropagation.
    Os pesos são ajustados apenas por Algoritmo Genético (AG).
    """

    def __init__(self, input_size=9, hidden_size=18, output_size=9):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Xavier simples
        limit1 = np.sqrt(6 / (input_size + hidden_size))
        self.W1 = np.random.uniform(-limit1, limit1, (hidden_size, input_size))
        self.b1 = np.zeros(hidden_size, dtype=float)

        limit2 = np.sqrt(6 / (hidden_size + output_size))
        self.W2 = np.random.uniform(-limit2, limit2, (output_size, hidden_size))
        self.b2 = np.zeros(output_size, dtype=float)

    # ------------------------------------------------------------------
    # Ferramentas para o Algoritmo Genético (cromossomo = todos os pesos)
    # ------------------------------------------------------------------
    @property
    def num_weights(self):
        return (
            self.W1.size + self.b1.size +
            self.W2.size + self.b2.size
        )

    def to_chromosome(self) -> np.ndarray:
        """
        Achata todos os pesos e biases em um único vetor 1D.
        """
        return np.concatenate([
            self.W1.flatten(),
            self.b1.flatten(),
            self.W2.flatten(),
            self.b2.flatten()
        ]).astype(float)

    def set_from_chromosome(self, chrom):
        """
        Atualiza W1, b1, W2, b2 a partir de um cromossomo 1D.
        """
        chrom = np.asarray(chrom, dtype=float)
        assert chrom.size == self.num_weights, (
            f"Tamanho do cromossomo inválido: {chrom.size}, esperado {self.num_weights}"
        )

        i = 0
        w1_size = self.W1.size
        b1_size = self.b1.size
        w2_size = self.W2.size
        b2_size = self.b2.size

        self.W1 = chrom[i:i + w1_size].reshape(self.W1.shape)
        i += w1_size

        self.b1 = chrom[i:i + b1_size]
        i += b1_size

        self.W2 = chrom[i:i + w2_size].reshape(self.W2.shape)
        i += w2_size

        self.b2 = chrom[i:i + b2_size]
        i += b2_size

    @classmethod
    def from_chromosome(cls, chrom, input_size=9, hidden_size=18, output_size=9):
        net = cls(input_size, hidden_size, output_size)
        net.set_from_chromosome(chrom)
        return net

    # ------------------------------------------------------------------
    # Forward + escolha de jogada
    # ------------------------------------------------------------------
    def _encode_board(self, board):
        """
        Converte tabuleiro 3x3 (com valores -1, 0, 1) em vetor 1D de 9 floats.
        """
        flat = []
        for linha in board:
            for v in linha:
                flat.append(float(v))
        return np.array(flat, dtype=float)

    def forward(self, board):
        """
        Propagação direta:
        board -> [9 saídas] (uma por casa do tabuleiro).
        """
        x = self._encode_board(board)          # shape: (9,)
        h = np.tanh(self.W1 @ x + self.b1)     # shape: (hidden_size,)
        z = self.W2 @ h + self.b2              # shape: (9,)

        # Softmax para interpretar como "preferência"
        exp_z = np.exp(z - np.max(z))
        y = exp_z / (exp_z.sum() + 1e-8)
        return y  # shape: (9,)

    def escolher_jogada(self, board, movimentos_validos):
        """
        Dado o tabuleiro e os movimentos válidos [(i,j),...],
        escolhe a casa que tem maior valor de saída entre as válidas.
        """
        output = self.forward(board)  # vetor com 9 scores
        indices_ordenados = np.argsort(output)[::-1]  # maior -> menor

        # mapa: índice (0..8) -> (linha,col)
        valid_indices = {3 * i + j: (i, j) for (i, j) in movimentos_validos}

        # tenta do melhor score até o pior
        for idx in indices_ordenados:
            if idx in valid_indices:
                return valid_indices[idx]

        # fallback de segurança
        if movimentos_validos:
            return movimentos_validos[0]
        return None
