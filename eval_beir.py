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
    beir_folder: str = "/data/davide/dragon/beir",
    model_ckpt = "/data/davide/models/t5/rankt5/flan-t5-small-difference-total-30/checkpoint-1/",
    n_docs = 100,
    score_strategy = "difference",
    corpus: str = "",
):

    # Set device to CUDA if available, else use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get scoring function based on user-defined strategy
    get_score = getattr(Score, score_strategy)
    # Create the output directory if it does not exist
    os.makedirs(outdir, exist_ok=True)

    # Load and initialize the T5 model and tokenizer for re-ranking
    print(f"===== LOADING  MonoT5: {model_ckpt} =====")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    # Set the model to evaluation mode
    model.eval()
    
    # if a specific corpus is indicated, eval only on it, otherwise on full benchmark
    if corpus in DATASET_NAMES:
        corpus_to_eval = [corpus]
    else:
        corpus_to_eval = DATASET_NAMES

    # Loop through each corpus in the defined dataset names
    for corpus in corpus_to_eval:
        print(f"====== {corpus} ======")

        # Load query and document data, and initial ranking from retriever
        runfile_tsv_path=f"{path_to_runfile}/{corpus}"
        qid2query, pid2doc = read_beir(beir_folder, corpus)
        qid2pid, qid2pidscore = read_trec(runfile_tsv_path)

        # Set path for the output reranked results
        trec_out_file_path=f"{outdir}/{corpus}"

        # Loop through each query and its associated documents
        for qid in tqdm(qid2pid, desc="running predictions"):
            query = qid2query[qid]
            pids = qid2pid[qid][:n_docs]
            texts = [pid2doc[pid] for pid in pids]

            # Initialize empty list to store scores
            scores = []
            for pid in pids:
                doc = pid2doc[pid]
                # Prepare model input
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
                
                # Pass the input through the model and get re-ranking scores
                with torch.no_grad():
                    output = model(
                        input_ids = input_ids.to(device), 
                        attention_mask = attention_mask.to(device), 
                        decoder_input_ids = decode_ids.to(device),    
                    ) 
                logits = output.logits[:,0,:]
                scores.append(get_score(logits).item())

            # Sort the documents by their scores in descending order
            reranked_pids = dict(zip(pids, scores))
            reranked_pids = dict(sorted(reranked_pids.items(), key=operator.itemgetter(1), reverse=True))

            # Write the reranked results to the output file
            with open(trec_out_file_path, 'a') as f:
                for i, (pid, score) in enumerate(reranked_pids.items()):
                    f.write(f"{qid} Q0 {pid} {i+1} {score} TWOLAR\n")


if __name__ == "__main__":
    CLI(main)
        
