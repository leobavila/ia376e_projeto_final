import argparse
import collections
import torch
from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import DuoT5
from transformers import T5ForConditionalGeneration
from tqdm import tqdm
from typing import List

import time


def load_run(path):
    """Loads run into a dict of key: query_id, value: list of candidate doc
    ids."""

    # We want to preserve the order of runs so we can pair the run file with
    # the TFRecord file.
    print('Loading run...')
    run = collections.OrderedDict()
    with open(path) as f:
        for line in tqdm(f):
            query_id, _, doc_title, rank, _, _ = line.split()
            if query_id not in run:
                run[query_id] = []
            run[query_id].append((doc_title, int(rank)))

    # Sort candidate docs by rank.
    print('Sorting candidate docs by rank...')
    sorted_run = collections.OrderedDict()
    for query_id, doc_titles_ranks in tqdm(run.items()):
        doc_titles_ranks.sort(key=lambda x: x[1])
        doc_titles = [doc_titles for doc_titles, _ in doc_titles_ranks]
        sorted_run[query_id] = doc_titles

    return sorted_run


parser = argparse.ArgumentParser(description='Run inference with duoT5.')
parser.add_argument('--collection', required=True, type=str, help='Path containing the MS MARCO collection.')
parser.add_argument('--topics', required=True, type=str, help='Path containing topics (queries).')
parser.add_argument('--input_run', required=True, type=str, help='Path to the TREC-formatted run file to be reranked.')
parser.add_argument('--output_run', required=True, type=str, help='Path to write the TREC-formatted run file.')
parser.add_argument('--num_rerank', type=int, default=30, help='Number of documents to be reranked.')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = T5ForConditionalGeneration.from_pretrained('castorini/duot5-base-msmarco').to(device).eval()
reranker = DuoT5(model=model)
print(f'Running on {device}')

queries = {}
with open(args.topics) as f:
    for line in f:
        query_id, query_text = line.strip().split('\t')
        queries[query_id] = query_text

collection = {}
with open(args.collection) as f:
    for line in f:
        doc_id, doc_text = line.strip().split('\t')
        collection[doc_id] = doc_text

run = load_run(path=args.input_run)

with open(args.output_run, 'w') as fout:
    for query_id, doc_ids in tqdm(run.items(), total=len(run)):
        
        query = Query(queries[query_id])
        doc_ids = doc_ids[:args.num_rerank]
        texts = [Text(collection[doc_id], {'docid': doc_id}, 0) for doc_id in doc_ids]
        reranked = reranker.rerank(query, texts)
        for rank, doc in enumerate(reranked, start=1): 
            fout.write(f'{query_id} Q0 {doc.metadata["docid"]} {rank} {doc.score} duo\n')
print('Done!')
