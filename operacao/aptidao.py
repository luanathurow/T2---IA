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


def jogar_uma_partida(rn, partida_index: int, num_partidas: int, modo_minimax: str, verbose: bool):
	"""Executa uma única partida entre `rn` e Minimax.

	Retorna (resultado, acertos, erros).
	"""
	game = TicTacToe()

	# alterna quem começa de forma determinística: RN começa nas partidas pares
	rn_jogador = 1 if (partida_index % 2 == 0) else -1
	current = rn_jogador

	acertos = 0
	erros = 0

	if verbose:
		print(f"Partida {partida_index+1}/{num_partidas}: RN joga como {rn_jogador}")

	while not game.jogo_terminou():
		if current == rn_jogador:
			tab_flat = [game.board[i][j] for i in range(3) for j in range(3)]
			movs = game.movimentos_disponiveis()
			if not movs:
				break

			escolha = rn.escolher_jogada(tab_flat, movs)
			if escolha is None or escolha not in movs:
				erros += 1
				if verbose:
					print("  Jogada inválida/ilegal pela RN — conta como erro")
				return -1, acertos, erros

			ok = game.fazer_jogada(escolha[0], escolha[1], current)
			if not ok:
				erros += 1
				if verbose:
					print("  Jogada ilegal (casa ocupada) pela RN — conta como erro")
				return -1, acertos, erros

			acertos += 1
		else:
			mov = melhor_jogada_modo(game, current, modo=modo_minimax)
			if mov is None:
				break
			game.fazer_jogada(mov[0], mov[1], current)

		current = -current

	vencedor = game.checar_vencedor()
	if vencedor is None:
		resultado = 0
	elif vencedor == rn_jogador:
		resultado = 1
	else:
		resultado = -1

	return resultado, acertos, erros


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

	total_score = 0.0

	# pesos para combinar componentes: resultado do jogo tem maior peso
	WEIGHT_RESULT = 1.0
	WEIGHT_ACERTOS = 0.1
	WEIGHT_ERROS = 0.2

	for partida in range(num_partidas):
		resultado, acertos, erros = jogar_uma_partida(rn, partida, num_partidas, modo_minimax, verbose)
		partida_score = WEIGHT_RESULT * resultado + WEIGHT_ACERTOS * acertos - WEIGHT_ERROS * erros
		total_score += partida_score

		if verbose:
			if resultado == 0:
				outcome = "empate"
			elif resultado == 1:
				outcome = "vitoria"
			else:
				outcome = "derrota"
			print(f"  Resultado: {outcome} | acertos={acertos} erros={erros} | partida_score={partida_score:.3f}")

	# retorna média por partida
	return total_score / num_partidas
