"""Função de aptidão para avaliar um vetor de pesos (cromossomo).

A função `aptidao` cria uma `RedeNeural` a partir do vetor de pesos,
faz o número configurável de partidas contra o Minimax
(usando `operacao.minimax.melhor_jogada_modo`) e retorna uma pontuação
agregada (+1 vitória da RN, 0 empate, -1 derrota).
"""

import random
from typing import Sequence

from tic_tac_toe import TicTacToe

try:
	# relative/absolute import fallback for rede_neural
	from rede_neural import RedeNeural
except Exception:
	from ..rede_neural import RedeNeural

try:
	from operacao.minimax import melhor_jogada_modo
except Exception:
	from .minimax import melhor_jogada_modo


def aptidao(pesos: Sequence[float], num_partidas: int = 20, modo_minimax: str = "dificil", verbose: bool = False) -> float:
	"""Avalia `pesos` jogando `num_partidas` contra o Minimax.

	- `pesos`: sequência de 180 floats (cromossomo)
	- `num_partidas`: quantas partidas jogar (padrão 20)
	- `modo_minimax`: modo passado para `melhor_jogada_modo` ('dificil'|'medio')
	- `verbose`: imprime progresso (opcional)

	Retorna a soma das pontuações: +1 por vitória da RN, 0 empate, -1 derrota.
	Valores maiores indicam indivíduos melhores.
	"""

	rn = RedeNeural(pesos)

	score = 0.0

	for partida in range(num_partidas):
		game = TicTacToe()

		# Define aleatoriamente quem começa: 1 (X) ou -1 (O)
		current = random.choice([1, -1])

		# Define qual jogador é a RN nesta partida
		rn_jogador = current

		if verbose:
			print(f"Partida {partida+1}/{num_partidas}: RN joga como {rn_jogador}")

		# Loop de jogo
		while not game.jogo_terminou():
			if current == rn_jogador:
				# Jogada da Rede Neural
				# prepara tabuleiro flat (linha-major)
				tab_flat = [game.board[i][j] for i in range(3) for j in range(3)]

				movs = game.movimentos_disponiveis()
				if not movs:
					break

				escolha = rn.escolher_jogada(tab_flat, movs)
				if escolha is None:
					# sem movimento válido
					break

				game.fazer_jogada(escolha[0], escolha[1], current)
			else:
				# Jogada do Minimax
				mov = melhor_jogada_modo(game, current, modo=modo_minimax)
				if mov is None:
					break
				game.fazer_jogada(mov[0], mov[1], current)

			# alterna jogador
			current = -current

		vencedor = game.checar_vencedor()
		if vencedor is None:
			# empate
			resultado = 0
		elif vencedor == rn_jogador:
			resultado = 1
		else:
			resultado = -1

		score += resultado

		if verbose:
			outcome = "empate" if resultado == 0 else ("vitoria" if resultado == 1 else "derrota")
			print(f"  Resultado: {outcome}")

	return score
