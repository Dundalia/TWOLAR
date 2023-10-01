import pandas as pd
import os
from pyserini.search.lucene import LuceneSearcher
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
from utils import Score, read_corpus
from jsonargparse import CLI
import torch
import operator

def main(
    trecdl_path: str, 
    index_path: str, 
    corpus_path: str,
    model_ckpt: str, 
    outdir: str, 
    score_strategy: str, 
    topk: int,
):

    # Determine device (either CPU or GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load MSMARCO corpus
    pid2doc = read_corpus(corpus_path)

    # Initialize pyserini bm25 Searcher 
    searcher = LuceneSearcher(index_path)

    # Load model and tokenizer
    print(f"===== LOADING  MonoT5: {model_ckpt} =====")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = model.eval()

    # Get the scoring method based on the strategy
    get_score = getattr(Score, score_strategy)

    # Evaluate the model on TREC-DL for each year (2019 and 2020)
    for year in ["2019", "2020"]:
        print(f"=== Evaluating TREC-DL {year} ===")

        # Load qrels and queries
        qrels = pd.read_csv(os.path.join(trecdl_path, f"{year}qrels-pass.txt"), 
                            sep=" ", names=["qid", "Q0", "pid", "score"])
        queries = pd.read_csv(os.path.join(trecdl_path, f"msmarco-test{year}-queries.tsv"),
                              sep="\t", names = ["qid", "text"])   

        # Filter out queries that are not in the qrels
        queries = queries[queries.qid.isin(qrels.qid)]
        
        # Prepare the output file path
        trec_out_file_path = os.path.join(outdir, f"trec-dl-{year}")

        # Iterate over each query
        for i, row in tqdm(
            queries.iterrows(), 
            desc="running predictions",
            total = len(queries)
        ):
            qid = str(row.qid)
            query = row["text"]
            hit = searcher.search(query, topk)
            pids = [sample.docid for sample in hit]
        
            scores = []
            for pid in pids:
                doc = pid2doc[pid]
                input = [f"Query: {query} Document: {doc} Relevant: "]

                # Tokenize the input for the model
                features = tokenizer(
                    input, truncation = True, 
                    return_tensors = "pt", max_length = 500, 
                    padding = True,
                )

                # Prepare input tensors
                input_ids = features.input_ids
                attention_mask = features.attention_mask
                decode_ids = torch.full((input_ids.size(0), 1),
                                         model.config.decoder_start_token_id,
                                         dtype=torch.long)

                # Inference step
                with torch.no_grad():
                    output = model(
                        input_ids = input_ids.to(device), 
                        attention_mask = attention_mask.to(device), 
                        decoder_input_ids = decode_ids.to(device),    
                    ) 
                logits = output.logits[:,0,:]
                scores.append(get_score(logits).item())

            # Rerank documents based on their scores
            reranked_pids = dict(zip(pids, scores))
            reranked_pids = dict(sorted(reranked_pids.items(), key=operator.itemgetter(1), reverse=True))
            
            # Save the reranked results to the output file
            with open(trec_out_file_path, 'a') as f:
                for i, (pid, score) in enumerate(reranked_pids.items()):
                    f.write(f"{qid} Q0 {pid} {i+1} {score} twolar\n")

        
        

if __name__ == "__main__":
    CLI(main)
