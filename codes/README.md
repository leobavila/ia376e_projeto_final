## Install pygaggle from git
pip install git+https://github.com/castorini/pygaggle.git


## Data Download

collection.tsv:
wget https://www.dropbox.com/s/m1n2wf80l1lb9j1/collection.tar.gz?dl=1

topics.dl20.txt:
https://raw.githubusercontent.com/castorini/anserini/master/src/main/resources/topics-and-qrels/topics.dl20.txt

qrels.dl20-passage.txt:
wget https://raw.githubusercontent.com/castorini/anserini/master/src/main/resources/topics-and-qrels/qrels.dl20-passage.txt


## Rerank
```
nohup python rerank.py \
    --collection=./collection.tsv \
    --topics=./topics.dl20.txt \
    --input_run=./base.dl20.p.dTq.rm3.mono.trec \
    --output_run=./base.dl20.p.dTq.rm3.duo.trec \
    --num_rerank=300 &

tail -100f nohup.out
```


## Evaluate
```
python -m pyserini.eval.trec_eval \
    -c \
    -l 2 \
    -m ndcg_cut.10,20 \
    -m map \
    -m P.20,30 \
    -m recall.100,1000 \
    -m recip_rank \
    ./qrels.dl20-passage.txt \
    ./base.dl20.p.dTq.rm3.duo.trec


python measure_judged.py \
    --qrels=./qrels.dl20-passage.txt \
    --run=./base.dl20.p.dTq.rm3.duo.trec \
    --cutoffs 10 20 \
    --topics-in-qrels-only
```
