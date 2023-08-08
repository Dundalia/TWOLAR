from utils import *
#from pyserini.search.lucene import LuceneSearcher
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, AutoModelForMaskedLM
#from pygaggle.rerank.transformer import MonoT5
#from pygaggle.rerank.base import Query, Text
import torch
#import pytrec_eval
from jsonargparse.cli import CLI
import mmap
from collections import defaultdict
import operator

all = True

DATASET_NAMES = [
    "trec-covid",
    "scifact",
    "nfcorpus",
    "dbpedia-entity",
    "webis-touche2020",
    'robust04',
    'signal1m',
    'trec-news'
]

if all:
    DATASET_NAMES = [
        "fiqa",
        "trec-covid",
        "scifact",
        "nfcorpus",
        "scidocs",
        "webis-touche2020",
        "dbpedia-entity",
        "arguana",
        "nq",
        "quora",
        "fever",
        "climate-fever",
        "hotpotqa",
        'cqadupstack/gaming',
        'cqadupstack/tex',
        'cqadupstack/programmers',
        'cqadupstack/wordpress',
        'cqadupstack/webmasters',
        'cqadupstack/mathematica',
        'cqadupstack/stats',
        'cqadupstack/gis',
        'cqadupstack/english',
        'cqadupstack/physics',
        'cqadupstack/unix',
        'cqadupstack/android',
        'bioasq',
        'robust04',
        'signal1m',
        'trec-news'
    ]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bm25_ndcg = []

score_type = "difference"
def get_score(logits):
    if score_type == "extra_id":
        return logits[:, 32089]
    if score_type == "difference":
        true_logits = logits[:, 1176]
        false_logits = logits[:, 6136]
        return true_logits - false_logits
    if score_type == "softmax":
        true_logits = logits[:, 1176]
        false_logits = logits[:, 6136]
        scores = [torch.nn.functional.log_softmax(torch.stack([true_logit, false_logit]), dim=0)[0]
                 for true_logit, false_logit in zip(true_logits, false_logits)]
        return torch.stack(scores) 

        
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

def main(
    outdir: str = "/data/davide/dragon/results/0804/bm25-flan-t5-xl-difference-30-ga-32/",
    path_to_runfile: str = "/data/davide/dragon/results/retriever/bm25",
    beir_folder: str = "/data/davide/dragon/beir",
    model_ckpt = "/data/davide/models/t5/rankt5/flan-t5-xl-difference-total-30-ga-32/checkpoint-1/",
    n_docs = 100,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(outdir, exist_ok=True)
    
    print(f"===== LOADING  MonoT5: {model_ckpt} =====")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model.eval()

    
    for corpus in DATASET_NAMES:
    
        print(f"====== {corpus} ======")

        ### import queries, corpus adn bm25 trec
        questions_tsv_path=f"{beir_folder}/{corpus}/queries.test.tsv"
        passages_tsv_path=f"{beir_folder}/{corpus}/collection.tsv"
        runfile_tsv_path=f"{path_to_runfile}/{corpus}"
        trec_out_file_path=f"{outdir}/{corpus}"
        
    
        ctxs = CSVDataset(passages_tsv_path)
        questions = QueryTSVDataset(questions_tsv_path)    
        qid2query = {}
        for i in range(len(questions)):
            qid2query[questions[i]["id"]] = questions[i]["question"]
        pid2doc = {}
        for i in range(len(ctxs)):
            if ctxs[i]["title"] != "": ## appending the title to the text
                pid2doc[ctxs[i]["id"]] = " ".join([ctxs[i]["title"],  ctxs[i]["text"]])
            else:
                pid2doc[ctxs[i]["id"]] = ctxs[i]["text"]      
        qid2pid, qid2pidscore = read_trec(runfile_tsv_path)
        ###
        
            
        for qid in tqdm(
            qid2pid, 
            desc="running predictions"
        ):
            query = qid2query[qid]
            pids = qid2pid[qid][:n_docs]
            texts = [pid2doc[pid] for pid in pids]

            scores = []
            for pid in pids:
                doc = pid2doc[pid]
                input = [f"Query: {query} Document: {doc} Relevant: "]
                features = tokenizer(
                    input, truncation = True, 
                    return_tensors = "pt", max_length = 500, 
                    padding = True,
                )
                input_ids = features.input_ids
                attention_mask = features.attention_mask
                decode_ids = torch.full((input_ids.size(0), 1),
                                         model.config.decoder_start_token_id,
                                         dtype=torch.long)
                with torch.no_grad():
                    output = model(
                        input_ids = input_ids.to(device), 
                        attention_mask = attention_mask.to(device), 
                        decoder_input_ids = decode_ids.to(device),    
                    ) 
                logits = output.logits[:,0,:]
                scores.append(get_score(logits).item())
                
            reranked_pids = dict(zip(pids, scores))
            reranked_pids = dict(sorted(reranked_pids.items(), key=operator.itemgetter(1), reverse=True))
            with open(trec_out_file_path, 'a') as f:
                for i, (pid, score) in enumerate(reranked_pids.items()):
                    f.write(f"{qid} Q0 {pid} {i+1} {score} Davide-NII-1\n")


if __name__ == "__main__":
    CLI(main)
        
