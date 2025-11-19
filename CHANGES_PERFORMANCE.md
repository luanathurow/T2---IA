## Principais mudanças que melhoraram desempenho

Este arquivo resume, de forma concisa, as alterações feitas no repositório que tiveram impacto direto no desempenho do treino e avaliação da Rede Neural.

1) Avaliação de aptidão concentrada no Minimax
- Implementada a função `operacao/aptidao.py` que avalia cromossomos jogando contra o Minimax e aplica penalizações por jogadas inválidas.
- Centralizar a lógica de avaliação permitiu medir fitness de forma consistente e alterar facilmente o número de partidas por indivíduo (`--games`).

2) Otimização do Minimax
- Substituído o Minimax bruto por uma versão com poda alpha-beta e um cache de transposição por chamada (`minimax.py`).
- Resultado: redução drástica do tempo de avaliação por partida, tornando viável treinar populações maiores e mais gerações.

3) Parale­lização das avaliações
- Adicionado suporte a `multiprocessing` em `AG.py` (opção `use_parallel`) para avaliar cromossomos em paralelo.
- `treino_ag.py` ativa paralelização por padrão; isso escala bem em máquinas com múltiplos núcleos e reduz o tempo total do treino.

4) Ajustes nos operadores do AG e parâmetros
- Parâmetros tornaram-se configuráveis: `elite`, `torneio`, `taxa_mut` e `std_mut` — permitiu experimentar reduzindo pressão de seleção e aumentando mutação para preservar diversidade.
- Ex.: reduzir `elite` e `torneio` e aumentar `taxa_mut` ajudou a evitar convergência prematura em diversos testes.

5) Scripts de automação e avaliação
- `treino_ag.py` (CLI) e `evaluate_vs_minimax.py` (avaliação em lote) automatizaram execuções, facilitando testes comparativos e coleta de estatísticas (e.g., 200 partidas por avaliação).

6) Interface e robustez
- `main.py` recebeu melhoria na entrada humana (agora 0-based) e pausas para revisar resultados, reduzindo erro humano durante testes e replays.

7) Estratégia de tuning aplicada
- Aumentar `--games` (jogos por indivíduo) reduz ruído no cálculo do fitness e produz avaliações mais estáveis.
- A combinação de população maior (`--pop`), mais gerações (`--generations`) e paralelização foi essencial para melhorar resultados sem aumento prohibitivo do tempo.

Notas operacionais rápidas
- Para reproduzir os treinos que melhoraram desempenho, use uma combinação como:
  - `python treino_ag.py --generations 100 --pop 80 --games 12 --elite 2 --torneio 2 --taxa_mut 0.12 --std_mut 0.12 --out ag_tuned.csv`
- Após o treino, avalie com: `python evaluate_vs_minimax.py -g 200 -m medio` e `-m dificil`.

Próximos passos (se quiser aprimorar mais)
- Implementar critério de parada com "patience" em vez de parar só por baixa variância;
- Salvar stats por geração (CSV) para análise de convergência e geração de plots;
- Paralelizar ainda mais (workers por GPU/BLAS ou usar job que execute várias sementes em paralelo) — envolve mais infraestrutura.

