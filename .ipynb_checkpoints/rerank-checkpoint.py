from utils import DATASET_NAMES, Score, read_trec, read_beir
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from jsonargparse import CLI
import mmap
from collections import defaultdict
import operator
import os
from tqdm import tqdm


def main(
    outdir: str = "/data/davide/dragon/results/developer/flan-t5-small",
    path_to_runfile: str = "/data/davide/dragon/results/retriever/bm25",
    beir_folder: str = "./beir",
    model_ckpt = "/data/davide/models/t5/rankt5/flan-t5-small-difference-total-30/checkpoint-1/",
    n_docs = 100,
    score_strategy = "difference",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    get_score = getattr(Score, score_strategy)
    os.makedirs(outdir, exist_ok=True)
    
    print(f"===== LOADING  MonoT5: {model_ckpt} =====")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model.eval()

    for corpus in ["nfcorpus"]:
    
        print(f"====== {corpus} ======")

        ### import queries, corpus adn bm25 trec
        runfile_tsv_path=f"{path_to_runfile}/{corpus}"
        qid2query, pid2doc = read_beir(beir_folder, corpus)
        qid2pid, qid2pidscore = read_trec(runfile_tsv_path)
        
        trec_out_file_path=f"{outdir}/{corpus}"
            
        for qid in tqdm( qid2pid, desc="running predictions"):
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
                    f.write(f"{qid} Q0 {pid} {i+1} {score} TWOLAR\n")


if __name__ == "__main__":
    CLI(main)
        
