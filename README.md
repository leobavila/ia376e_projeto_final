# ia376e_projeto_final

## Resumo:

O DuoT5 é um dos melhores reranqueadores que existe, porém, não funciona bem quando tem que reranquear muitos textos. O problema está possivelmente em sua função de agregação, onde, dado dois textos i e j, sendo i um texto relevante e j um texto não relevante mas sobre um assunto parecido, a chance de j ter um score maior que i ao acaso aumenta com o tamanho da lista de textos a serem reranqueados.

O objetivo desse trabalho é explorar diferentes funções de agregação a fim de reduzir tal efeito ao se aumentar o número de textos. Conseguimos obter melhores resultados para NDCG@10 e NDCG@20 dos que o previamente reportado e, além disso, encontramos funções que apresentaram característica monotônica a medida que se aumenta o número de textos candidatos.

## Dataset TREC 2020 Deep Learning Track

collection.tsv:
wget https://www.dropbox.com/s/m1n2wf80l1lb9j1/collection.tar.gz?dl=1

topics.dl20.txt:
https://raw.githubusercontent.com/castorini/anserini/master/src/main/resources/topics-and-qrels/topics.dl20.txt

qrels.dl20-passage.txt:
wget https://raw.githubusercontent.com/castorini/anserini/master/src/main/resources/topics-and-qrels/qrels.dl20-passage.txt

## Códigos:

[Multi_stage_reranker.ipynb](https://github.com/leobavila/ia376e_projeto_final/blob/main/codes/Multi_stage_reranker.ipynb):

Utilizado para executar o duoT5, disponível em uma [versão modificada](https://github.com/pedrogengo/pygaggle) do pygaggle, para reranquear a [saída obtida anteriormente](https://github.com/leobavila/ia376e_projeto_final/blob/main/results/base.dl20.p.dTq.rm3.mono.trec) do monoT5. O duoT5 gera a sua saída reranqueada em um formato trec, ele utiliza a função de agregação sym_sum para consolidar os pairwise_scores e gerar os pointwise_scores. A modificação do pacote permite obter os pairwise_scores durante o processo e, assim, é possível testar outras funções de agregação. O numero de candidatos a ser reranqueados pelo duoT5 pode ser configurado através da variável: num_rerank.

O resultado em formato trec, tanto do monoT5 quanto do duoT5, pode ser utilizado para gerar as métricas de avaliação (MRR, nDCG, Precision, Recall) através do pacote pyserini que também está presente no código. O código [measure_judged.py](https://github.com/leobavila/ia376e_projeto_final/blob/main/codes/measure_judged.py) é executado para gerar as métricas Judged.

O Multi Stage Reranker foi utilizado para gerar os pairwise_scores do duoT5 para 30, 50 e 100 documentos. Para gerar os pairwise scores de 300 candidatos, foi necessário alterar o código para reranquear 27 e 27 queries separadamente (duração de aproximadamente 8 horas cada um), visto que o Google Colab Pro possui uma limitação de tempo de execução e derruba o notebook.

[Multi_stage_reranker_300_candidates_part1.ipynb](https://github.com/leobavila/ia376e_projeto_final/blob/main/codes/Multi_stage_reranker_300_candidates_part1.ipynb) e [Multi_stage_reranker_300_candidates_part2.ipynb](https://github.com/leobavila/ia376e_projeto_final/blob/main/codes/Multi_stage_reranker_300_candidates_part2.ipynb):

Códigos para reranquear as 54 queries utilizando o duoT5 e 300 candidatos. São geradas duas partes tanto para o arquivo trec quanto para o arquivo json contendo os pairwise_scores que precisam ser unificadas posteriormente.

[Merge_300_candidates_part1_part2.ipynb](https://github.com/leobavila/ia376e_projeto_final/blob/main/codes/Merge_300_candidates_part1_part2.ipynb):

Código para unificar as partes 1 e 2 dos arquivos trec e json que são as saídas do duoT5 para 300 candidatos.

[Pairwise_Scores_to_TREC_Refactored_flips.ipynb](https://github.com/leobavila/ia376e_projeto_final/blob/main/codes/Pairwise_Scores_to_TREC_Refactored_flips.ipynb)

Código que utiliza os pairwise_scores em formato json como input. Possui a classe duoT5Aggregation(), criada para:

* tornar mais eficiente os testes de diferentes funções de agregação.
* permitir a comparação com o paper do Expando-Mono-Duo
* receber os pairwise_scores e uma função de agregação e gerar os respectivos pointwise_scores
* transformra os pointwise_scores no formato TREC para avaliação
* gerar todas as métricas a serem avaliadas (MAP, MRR, Precision, Recall, NDCG, Judged)
* fazer todo esse processo para diferentes números de candidatos por query (30, 50, 100, 300)

Além disso, fizemos também algumas análises de diferenças de score que o duoT5 gera quando comparamos os textos i e j e depois j e i para uma determinada query. Aqui no projeto chamamos de "flip" o evento em que o modelo "mudou de opinião" quanto ao texto mais importante para uma query.

## Pipeline:
Multi-Stage Ranking Architecture.

![image](https://user-images.githubusercontent.com/35712949/139275242-e37844b2-a8ed-4257-93e6-080d567ba6c1.png)

## Links de Suporte:

* Pygaggle: https://github.com/castorini/pygaggle
* Castorini: https://github.com/castorini
* Anserini: https://github.com/castorini/anserini
* Pyserini: https://github.com/castorini/pyserini
* Rerankers: https://github.com/castorini/pygaggle/blob/master/pygaggle/rerank/transformer.py

## Referências:

1) MonoBERT: https://arxiv.org/abs/1901.04085
2) Multi-Stage-Ranking: https://arxiv.org/abs/1910.14424
3) MonoT5:  https://arxiv.org/abs/2003.06713
4) BERT and Beyond: https://arxiv.org/abs/2010.06467
5) Expando-Mono-Duo: https://arxiv.org/abs/2101.05667
6) TREC-DL-20: https://arxiv.org/pdf/2102.07662.pdf

## Gits auxiliares:

* Pygaggle: https://github.com/castorini/pygaggle/
* Castorini: https://github.com/castorini
* Anserini: https://github.com/castorini/anserini
* Pyserini: https://github.com/castorini/pyserini
* Pygaggle modificado para retornar os pairwise scores: https://github.com/pedrogengo/pygaggle

## Modelos:

MonoT5: This model is a T5-base reranker fine-tuned on the MS MARCO passage dataset for 100k steps (or 10 epochs).<br />
Link: https://huggingface.co/castorini/monot5-base-msmarco

DuoT5: This model is a T5-base pairwise reranker fine-tuned on MS MARCO passage dataset for 50k steps (or 5 epochs).<br />
Link: https://huggingface.co/castorini/duot5-base-msmarco





