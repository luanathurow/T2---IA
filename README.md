# Projeto - Jogo da Velha (IA)

Resumo rápido
- Repositório implementa uma Rede Neural (MLP) treinada por Algoritmo Genético (AG) para jogar Jogo da Velha, avaliando aptidão contra um adversário Minimax.
- Inclui scripts para treinar (`treino_ag.py`), avaliar (`evaluate_vs_minimax.py`) e jogar interativamente (`main.py`).

Estrutura principal (resumida)
- `main.py`: interface CLI para jogar (humano vs humano / aleatório / Minimax / Rede treinada).
- `AG.py`: implementação do fluxo do Algoritmo Genético (inicialização, seleção, cruzamento, mutação). Chama `operacao.aptidao.aptidao` para avaliar cromossomos.
- `treino_ag.py`: script CLI que chama `treinar_ag(...)` com parâmetros configuráveis e salva um CSV resumo.
- `evaluate_vs_minimax.py`: roda N partidas entre a rede (carregada de `best_chromosome.npy`) e o Minimax, imprime estatísticas.
- `rede_neural.py`: classe `NeuralNetwork` (to/from chromosome, forward, escolher_jogada).
- `minimax.py`: função `melhor_jogada_modo` com modos `medio` e `dificil` e implementação com poda alpha-beta + cache.
- `operacao/aptidao.py`: função `aptidao(chromosome, partidas, modo)` que avalia um cromossomo jogando contra Minimax e aplica penalizações.
- `tic_tac_toe.py`: representação do tabuleiro e utilitários do jogo.

Arquivos gerados
- `best_chromosome.npy`: cromossomo (vetor) do melhor indivíduo salvo após treino.
- `ag_*.csv` / `ag_results_*.csv`: arquivos resumo criados pelo `treino_ag.py` (quando usado).
- `eval_*.txt`: saídas de avaliações executadas com `evaluate_vs_minimax.py`.

Como executar (exemplos)
- Jogar interativamente:
  - `python main.py`  # roda menu interativo

- Treinar (CLI simples):
  - Exemplo rápido: `python treino_ag.py --generations 10 --pop 30 --games 3`
  - Treino maior (exemplos usados nas execuções):
    - `python treino_ag.py --generations 60 --pop 60 --games 8 --out ag_results.csv`
    - `python treino_ag.py --generations 100 --pop 80 --games 12 --elite 2 --torneio 2 --taxa_mut 0.12 --std_mut 0.12 --out ag_tuned.csv`
  - Flags importantes: `--generations`, `--pop`, `--games` (jogos por indivíduo), `--elite`, `--torneio`, `--taxa_mut`, `--std_mut`, `--out`.
  - `treino_ag.py` ativa `use_parallel=True` por padrão para aproveitar múltiplos núcleos.

- Avaliar rede (após treino):
  - `python evaluate_vs_minimax.py --games 200 --modo dificil`  # salva/mostra estatísticas
  - `python evaluate_vs_minimax.py -g 200 -m medio`  # exemplo: modo médio (50% aleatório)

- Notas sobre convergência e tuning
- O projeto tem um critério simples de parada que detecta "baixa variância de fitness" e encerra o treino. Em alguns experimentos isso resulta em parada precoce quando muitos indivíduos alcançam o mesmo valor (fitness discreto). Recomendações:
  - Aumentar `--games` para avaliações mais confiáveis (reduz ruído do fitness).
  - Reduzir elitismo (`--elite`) e torneio (`--torneio`) para preservar diversidade.
  - Aumentar mutação (`--taxa_mut`, `--std_mut`) se perceber perda de diversidade.
  - Implementar critério de "patience" (parar só se não houver melhoria do melhor fitness por N gerações) — posso adicionar isso se quiser.

Performance
- A avaliação de aptidão é a parte mais custosa (minimax). O projeto já tem otimizações (alpha-beta + cache) e suporte a avaliação paralela via `multiprocessing` para acelerar treinos grandes.

Resultados de referência (exemplos que rodamos)
- Treinos e avaliações que rodamos no workspace geraram arquivos como `ag_tuned.csv`, `ag_results_large.csv` e `eval_*.txt`. Exemplos de números observados nas execuções:
  - Contra Minimax `dificil`: muitas execuções resultaram em 0% vitórias da rede, mas alta fração de empates (ex.: 50% empates) — rede evita perder, mas não força vitória contra Minimax ótimo.
  - Contra Minimax `medio`: após tuning a rede chegou a ~40% vitórias em alguns testes.

Próximos passos sugeridos
- Se desejar, posso:
  - Implementar critério de `patience` para evitar paradas prematuras.
  - Adicionar logging CSV por geração com stats detalhados.
  - Gerar scripts de benchmark automatizados para comparar configurações.

Contato / execução
- Está tudo pronto para rodar localmente: veja os comandos acima. Se quiser que eu aplique alguma mudança (por exemplo `--seed` ou `patience`) ou gere um relatório/plot, diga qual das opções prefere.

---
Arquivo gerado automaticamente pelo assistente — resumo conciso das funcionalidades e instruções.
