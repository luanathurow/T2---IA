import argparse
import csv
import time
import numpy as np

from AG import treinar_ag


def main():
    parser = argparse.ArgumentParser(description='Treinar Rede com AG (script CLI)')
    parser.add_argument('--generations', '-g', type=int, default=50)
    parser.add_argument('--pop', '-p', type=int, default=40)
    parser.add_argument('--hidden', type=int, default=18)
    parser.add_argument('--modo', '-m', type=str, default='dificil', choices=['medio','dificil'])
    parser.add_argument('--games', type=int, default=5, help='Jogos por indivíduo na aptidão')
    parser.add_argument('--elite', type=int, default=4)
    parser.add_argument('--torneio', type=int, default=3)
    parser.add_argument('--taxa_mut', type=float, default=0.1)
    parser.add_argument('--std_mut', type=float, default=0.1)
    parser.add_argument('--out', type=str, default='ag_stats.csv')

    args = parser.parse_args()

    start = time.time()
    _, _ = treinar_ag(
        num_geracoes=args.generations,
        pop_size=args.pop,
        hidden_size=args.hidden,
        modo_minimax=args.modo,
        partidas_por_individuo=args.games,
        elite_size=args.elite,
        torneio_k=args.torneio,
        taxa_mutacao=args.taxa_mut,
        std_mutacao=args.std_mut,
        use_parallel=True,
        num_workers=None
    )

    elapsed = time.time() - start
    print(f"Treino finalizado em {elapsed:.1f}s. Melhor cromossomo salvo em 'best_chromosome.npy'.")

    # Gera CSV simples com uma linha de resumo
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['generations','pop','hidden','modo','games','elite','torneio','taxa_mut','std_mut','time_s'])
        w.writerow([args.generations, args.pop, args.hidden, args.modo, args.games, args.elite, args.torneio, args.taxa_mut, args.std_mut, f"{elapsed:.1f}"])

if __name__ == '__main__':
    main()
