import numpy as np
import random

from tic_tac_toe import TicTacToe
from rede_neural import NeuralNetwork
from minimax import melhor_jogada_modo


def aptidao(chromosome, partidas: int = 5, modo: str = 'dificil', hidden_size: int = 18) -> float:
	"""
	Avalia um cromossomo (vetor de pesos) jogando `partidas` partidas contra o Minimax.

	- A rede neural sempre começa jogando como X (1), conforme enunciado.
	- `modo` é repassado para `melhor_jogada_modo` ('medio'|'dificil').
	- Penaliza jogadas inválidas (escolher célula ocupada).

	Retorna o fitness médio por partida (float).
	"""
	chrom = np.asarray(chromosome, dtype=float)
	rede = NeuralNetwork.from_chromosome(chrom, hidden_size=hidden_size)

	total_score = 0.0

	for _ in range(partidas):
		jogo = TicTacToe()
		jogador_atual = 1  # Rede começa como X

		penalidade_invalidos = 0
		jogadas_validas = 0

		while not jogo.jogo_terminou():
			if jogador_atual == 1:
				movs_validos = jogo.movimentos_disponiveis()
				if not movs_validos:
					break

				l, c = rede.escolher_jogada(jogo.board, movs_validos)

				# Se escolher célula inválida, penaliza e força jogada válida
				if not jogo.jogada_valida(l, c):
					penalidade_invalidos += 1
					l, c = random.choice(movs_validos)

				jogo.fazer_jogada(l, c, 1)
				jogadas_validas += 1

			else:
				# Minimax como adversário
				move = melhor_jogada_modo(jogo, -1, modo)
				if move is None:
					movs = jogo.movimentos_disponiveis()
					if not movs:
						break
					l, c = random.choice(movs)
				else:
					l, c = move
				jogo.fazer_jogada(l, c, -1)

			jogador_atual *= -1

		vencedor = jogo.checar_vencedor()

		# pontuação por partida (valores ajustáveis)
		if vencedor == 1:
			total_score += 10
		elif vencedor == -1:
			total_score -= 8
		else:
			total_score += 5

		# penalizações / recompensas adicionais por jogadas
		total_score -= 3 * penalidade_invalidos
		total_score += 1 * jogadas_validas

	# normaliza por partidas
	return total_score / max(1, partidas)
