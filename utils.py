from datasets import load_dataset, load_from_disk
import pandas as pd
import numpy as np
import json
import tempfile
from tqdm import tqdm
from contextlib import contextmanager
import sys
import os
from io import StringIO
import random
import mmap
from collections import defaultdict
import operator
import torch

DATASET_NAMES = [
    "msmarco",
    "fiqa",
    "trec-covid",
    "scifact",
    "nfcorpus",
    "nq",
    "hotpotqa",
    "arguana",
    "webis-touche2020",
    "quora",
    "dbpedia-entity",
    "scidocs",
    "fever",
    "climate-fever",
]

InPars_MODEL_NAMES = {
    "fiqa": "fiqa",
    "trec-covid": "trec_covid",
    "scifact": "scifact",
    "nfcorpus": "nfcorpus",
    "msmarco": None,
    "nq": "nq",
    "hotpotqa": "hotpotqa",
    "arguana": "arguana",
    "webis-touche2020": "touche",
    "quora": "quora",
    "dbpedia-entity": "dbpedia",
    "scidocs": "scidocs",
    "fever": "fever",
    "climate-fever": "climate_fever",
    'cqadupstack/gaming': "cqadupstack-gaming" ,
    'cqadupstack/tex': "cqadupstack-tex",
    'cqadupstack/programmers': "cqadupstack-programmers",
    'cqadupstack/wordpress': "cqadupstack-wordpress",
    'cqadupstack/webmasters': "cqadupstack-webmasters",
    'cqadupstack/mathematica': "cqadupstack-mathematica",
    'cqadupstack/stats': "cqadupstack-stats",
    'cqadupstack/gis': "cqadupstack-gis",
    'cqadupstack/english': "cqadupstack-english",
    'cqadupstack/physics': "cqadupstack-physics",
    'cqadupstack/unix': "cqadupstack-unix",
    'cqadupstack/android': "cqadupstack-android",
    'bioasq': "bioasq",
    'robust04': "robust04",
    'signal1m': "signal",
    'trec-news': "trecnews"
}


###### PROBLEMS WITH ARGUANA ######

## The following indexes are in qrels["corpus-id"] but not in docs["_id"] ##

ARGUANA_DOCS_BAD_INDEXES = {'test-education-ufsdfkhbwu-con03b',
                            'test-free-speech-debate-yfsdfkhbwu-con03b',
                            'test-politics-dhwem-pro06b',
                            'test-science-sghwbdgmo-con03b',
                            'test-society-asfhwapg-con04b'}

ARGUANA_QUERIES_BAD_INDEXES = {'test-education-ufsdfkhbwu-con03a',
                               'test-free-speech-debate-yfsdfkhbwu-con03a',
                               'test-politics-dhwem-pro06a',
                               'test-science-sghwbdgmo-con03a',
                               'test-society-asfhwapg-con04a'}

## Arguana's queries are also documents !!! ##
## Arguana is about counterarguments, ill posed !! ##


##############################################################
## Context manager to suppress all the output of a function ##
##############################################################

@contextmanager
def suppress_all_output():
    # Store original values
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    # Redirect output to StringIO (in-memory text buffer)
    sys.stdout = StringIO()
    sys.stderr = StringIO()

    try:
        yield
    finally:
        # Restore original values
        sys.stdout = old_stdout
        sys.stderr = old_stderr

##################################################################
## Function to get the relevant documents ids given a query id  ##
##################################################################

def get_gold_ids(query_id, 
                 qrels):
  
    return list(qrels[(qrels['query-id']==query_id) &
                      (qrels['score']!=0)]['corpus-id'])

###############################################
## Function to generate #size hard triplets  ##
###############################################


def generate_random_triplets(size, 
                            docs, 
                            queries, 
                            qrels):
    
    rel_qrels = qrels[qrels["score"] != 0]
    size = min(size, len(rel_qrels))
    idxs = np.random.choice(range(len(rel_qrels)), size, replace=False)
    docs_id_set = set(docs['_id'])
    triplets = []    
    for i, row in tqdm(rel_qrels.iloc[idxs].iterrows(), 
                       total=size, 
                       desc="Generating Random Triplets", 
                       leave = False):
        query_id = row["query-id"]
        gold_id = row["corpus-id"]
        gold_ids = get_gold_ids(query_id, 
                                qrels)
        neg_id = random.choice(tuple(docs_id_set - set(gold_ids + [query_id])))
        triplets.append((query_id, gold_id, neg_id))
    return triplets

###################################################
## Function to get topk bm25 retrieved document  ##
###################################################

def bm25_retrieve(query_id, 
                  topk, 
                  docs, 
                  queries, 
                  qrels, 
                  searcher, 
                  include_relevants = True):
    
    rel_qrels = qrels[qrels["score"] != 0]
    query = queries[queries["_id"] == query_id].text.item()
    gold_ids = get_gold_ids(query_id, 
                            qrels)
    
    
    
    if include_relevants:
        hit = searcher.search(query, k=topk)
        docids = [sample.docid for sample in hit]
    else:
        hit = searcher.search(query, k=topk+5)
        docids = []
        for sample in hit:
                if sample.docid not in gold_ids + [query_id]:
                    docids.append(sample.docid)
                    if len(docids) == topk:
                        break
    
    return docids
        
###############################################
## Function to generate #size hard triplets  ##
###############################################

def generate_hard_triplets(size, 
                          docs, 
                          queries, 
                          qrels, 
                          searcher):
    rel_qrels = qrels[qrels["score"] != 0]
    size = min(size, len(rel_qrels))
    idxs = np.random.choice(range(len(rel_qrels)), size, replace=False)
    triplets = []
    for i, row in tqdm(rel_qrels.iloc[idxs].iterrows(), 
                       total=size, 
                       desc="Generating Hard Triplets", 
                       leave = False):
        
        query_id = row["query-id"]
        query = queries[queries["_id"] == query_id].text.item()

        gold_id = row["corpus-id"]
        gold_ids = get_gold_ids(query_id, 
                                qrels)
        
        hit = searcher.search(query, k=10)
        
        found = False
        for sample in hit:
            if sample.docid not in gold_ids + [query_id]:
                neg_id = sample.docid
                found = True
                break
        if not found:
            neg_id = random.choice(docs['_id'])
        triplets.append((query_id, gold_id, neg_id))
    return triplets


###############################################
## Function to generate #size hard ntuplets  ##
###############################################

def generate_hard_ntuples(n, 
                          size, 
                          docs, 
                          queries, 
                          qrels, 
                          searcher):
    rel_qrels = qrels[qrels["score"] != 0]
    size = min(size, len(rel_qrels))
    idxs = np.random.choice(range(len(rel_qrels)), size, replace=False)
    ntuplets = []
    for i, row in tqdm(rel_qrels.iloc[idxs].iterrows(), 
                       total=size, 
                       desc=f"Generating Hard {n}-tuplets", 
                       leave = False):
        
        query_id = row["query-id"]
        query = queries[queries["_id"] == query_id].text.item()

        gold_id = row["corpus-id"]
        gold_ids = get_gold_ids(query_id, 
                                qrels)
        
        hit = searcher.search(query, k=n+5)
        neg_ids = []
        for sample in hit:
            if sample.docid not in gold_ids + [query_id]:
                neg_ids.append (sample.docid)
                if len(neg_ids) == n:
                    break

        ntuplets.append((query_id, gold_id, neg_ids))
    return ntuplets

###################################################################
## Function to compute recall between gold and predicted indices ##
###################################################################

def recall(pred_ids, gold_ids):
    return len(set(gold_ids).intersection(set(pred_ids))) / len(gold_ids)

###################################################
## Function to download corpus queries and qrels ##
###################################################

def load_BeIR_dataset(dataset_name, 
                      corpus_from_disk = True, 
                      queries_from_disk = True,
                      qrels_from_disk = True, 
                      test_only = False):
    
    assert dataset_name in DATASET_NAMES, f"{dataset_name} is not a valid BeIR dataset name. \nThe valid names are {DATASET_NAMES}"
    
    # load corpus
    if corpus_from_disk:
        with open(f"/data/davide/beir/{dataset_name}_corpus_json/{dataset_name}_corpus.json") as json_file:
            data = json.load(json_file)
        docs = pd.DataFrame.from_dict(data).rename(columns = {'id':'_id', 'contents':'text'})
        docs['_id'] = docs['_id'].map(str)
    else: 
        corpus = load_dataset(f"BeIR/{dataset_name}", "corpus")
        docs = pd.DataFrame(corpus["corpus"]).drop(columns = ["title"])
    
    # load queries
    if queries_from_disk:
        queries = load_from_disk(f"/data/davide/beir/{dataset_name}-queries")
        queries = pd.DataFrame(queries["queries"]).drop(columns = ["title"])
    else:
        tempdir = tempfile.mkdtemp()
        queries = load_dataset(f"BeIR/{dataset_name}", "queries", cache_dir = tempdir)
        queries = pd.DataFrame(queries["queries"]).drop(columns = ["title"])
    
    # load qrels
    if qrels_from_disk:
        qrels_raw = load_from_disk(f"/data/davide/beir/{dataset_name}-qrels")
    else: 
        qrels_raw = load_dataset(f"BeIR/{dataset_name}-qrels")
    
    
    if test_only:
        qrels = pd.DataFrame(qrels_raw["test"])
    else:
        # concat the splits of qrels in a single DataFrame
        qrels = []
        for key in qrels_raw.keys():
            qrels.append(pd.DataFrame(qrels_raw[key]))
        qrels = pd.concat(qrels)
        
    qrels['query-id'] = qrels['query-id'].map(str)
    qrels['corpus-id'] = qrels['corpus-id'].map(str)

    ## Correct ARGUANA
    if dataset_name == "arguana":
        qrels = qrels[~qrels['corpus-id'].isin(ARGUANA_DOCS_BAD_INDEXES)]
        queries = queries[~queries['_id'].isin(ARGUANA_QUERIES_BAD_INDEXES)]
        docs = docs[~docs["_id"].isin(ARGUANA_DOCS_BAD_INDEXES)]

    return docs, queries, qrels

######################################################################
## Function to load qrels and save them locally in ./BeIR directory ##
######################################################################

def load_qrels_and_save_to_disk(dataset_name):
    
    assert dataset_name in DATASET_NAMES, f"{dataset_name} is not a valid BeIR dataset name. \nThe valid names are {DATASET_NAMES}"

    dataset_filename = f"/data/davide/beir/{dataset_name}-qrels"
    qrels = load_dataset(dataset_filename)
    qrels.save_to_disk(dataset_filename)
    
###############################################################
## Function to load dataset and transform it to pd.DataFrame ##
###############################################################

def load_and_preprocess_corpus(dataset_name):
    
    assert dataset_name in DATASET_NAMES, f"{dataset_name} is not a valid BeIR dataset name. \nThe valid names are {DATASET_NAMES}"

    corpus = load_dataset(f"BeIR/{dataset_name}", "corpus")
    docs = pd.DataFrame(corpus["corpus"]).drop(columns = ["title"])
    return docs

################################################
## Function to transform pd.DataFrame in Json ##
## as wanted by Pyserini and save it to disk  ##
################################################

def save_corpus_for_pyserini(docs, dataset_name):
    
    assert dataset_name in DATASET_NAMES, f"{dataset_name} is not a valid BeIR dataset name. \nThe valid names are {DATASET_NAMES}"
    
    doc_json = []

    for i, row in docs.iterrows():
        doc_json.append({
         "id": row["_id"],
         "contents": row["text"]   
        })

    with open(f'/data/davide/beir/{dataset_name}_corpus.json', 'w') as outfile:
        json.dump(doc_json, outfile)
        
        
        
def qrels_to_trec(qrels):
    
    trec_qrels = dict()
    for query_id in qrels["query-id"].unique():
        doc_ids = qrels[qrels['query-id'] == qid]['corpus-id']
        doc_scores = qrels[qrels['query-id'] == qid]['score']
        trec_qrels[query_id]=dict(zip(doc_ids, doc_scores))
       
    return trec_qrels







class MemoryMappedDataset(torch.utils.data.Dataset):
    """
    A memory mapped dataset.
    """

    def __init__(self, path, header=False):
        local_path = path ##
        self.file = open(local_path, mode="r")
        self.mm = mmap.mmap(self.file.fileno(), 0, prot=mmap.PROT_READ)
        if header:
            line = self.mm.readline()
        self.offset_dict = {0: self.mm.tell()}
        line = self.mm.readline()
        self.count = 0
        while line:
            self.count += 1
            offset = self.mm.tell()
            self.offset_dict[self.count] = offset
            line = self.mm.readline()

    def __len__(self):
        return self.count

    def process_line(self, line):
        return line

    def __getitem__(self, index):
        offset = self.offset_dict[index]
        self.mm.seek(offset)
        line = self.mm.readline()
        return self.process_line(line)

class CSVDataset(MemoryMappedDataset):
    """
    A memory mapped dataset for csv files
    """

    def __init__(self, path, sep="\t"):
        super().__init__(path, header=True)
        self.sep = sep
        self.columns = self._get_header()

    def _get_header(self):
        self.mm.seek(0)
        return self._parse_line(self.mm.readline())

    def _parse_line(self, line):
        """Implementation of csv quoting."""
        row = line.decode().rstrip("\r\n").split(self.sep)
        for i, val in enumerate(row):
            if val and val[0] == '"' and val[-1] == '"':
                row[i] = val.strip('"').replace('""', '"')
        return row

    def process_line(self, line):
        vals = self._parse_line(line)
        if len(self.columns) == len(vals):
            return dict(zip(self.columns, vals))
        else:  # hack
            self.__getitem__(0)

class QueryTSVDataset(MemoryMappedDataset):
    """
    A memory mapped dataset for query tsv files with the format qid\tquery_text\n
    """

    def __init__(self, path, sep="\t"):
        super().__init__(path, header=False)
        self.sep = sep

    def _parse_line(self, line):
        """Implementation of csv quoting."""
        row = line.decode().rstrip("\r\n").split(self.sep)
        for i, val in enumerate(row):
            if val and val[0] == '"' and val[-1] == '"':
                row[i] = val.strip('"').replace('""', '"')
        return row

    def process_line(self, line):
        vals = self._parse_line(line)
        return {
                "id": vals[0],
                "question": vals[1],
            }

def read_query(file):
    qid2query = {}
    with open(file, 'r') as fin:
        for i, line in tqdm(enumerate(fin), desc='read query file {}'.format(file)):
            qid, query = line.strip().split('\t')
            qid2query[qid] = query
    return qid2query


def read_corpus(corpus_path):
    pid2doc = {}
    with open(corpus_path, 'r') as fin:
        for i, line in tqdm(enumerate(fin), desc='read corpus file {}'.format(corpus_path)):
            pid, doc, title = line.strip().split('\t')
            pid2doc[pid] = doc
    return pid2doc
    

def read_trec(trec_in_path):
    qid2pid = defaultdict(list)
    qid2pidscore = defaultdict(list)
    with open(trec_in_path, 'r') as fin:
        for line in tqdm(fin, desc = 'read trec file {}'.format(trec_in_path)):
            qid, _, pid, rank, score, _= line.strip().split(' ')
            qid2pid[qid].append(pid)
            qid2pidscore[qid].append(float(score))
    return qid2pid, qid2pidscore


def read_beir(corpus):

    beir_folder = "/data/davide/dragon/beir"
    questions_tsv_path=f"{beir_folder}/{corpus}/queries.test.tsv"
    passages_tsv_path=f"{beir_folder}/{corpus}/collection.tsv"

    ctxs = CSVDataset(passages_tsv_path)
    questions = QueryTSVDataset(questions_tsv_path)    
    qid2query = {}
    for i in range(len(questions)):
        qid2query[questions[i]["id"]] = questions[i]["question"]
    pid2doc = {}
    for i in range(len(ctxs)):
        if ctxs[i] is not None:
            if ctxs[i]["title"] != "": ## appending the title to the text
                pid2doc[ctxs[i]["id"]] = " ".join([ctxs[i]["title"],  ctxs[i]["text"]])
            else:
                pid2doc[ctxs[i]["id"]] = ctxs[i]["text"]   
    
    return qid2query, pid2doc
    
