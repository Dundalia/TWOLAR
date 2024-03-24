# TWOLAR: a TWO-steps LLM-Augmented distillation method for passage Reranking

This repository provides the implementation of the paper "TWOLAR: a TWO-steps LLM-Augmented distillation method for passage Reranking". The TWOLAR method introduces a novel approach to passage reranking, leveraging two-step distillation in combination with language model augmentation to improve results in various reranking benchmarks.

![data-generation](/images/data-generation.png)
*Illustration of the methodology adopted to build the distillation dataset.*

## Table of Contents
- [A quick example](#a-quick-example)
- [Training](#training)
- [Evaluation](#evaluation)
  - [Run files format](#run-files-format)   
  - [Evaluating on BEIR](#evaluating-on-beir)
  - [Evaluating on TREC-DL2019 and TREC-DL2020](#evaluating-on-trec-dl2019-and-trec-dl2020)

# A quick example
Here we explain how to utilize the models.  
First of all, define the model checkpoint and the corresponding score strategy:
```python
from utils import Score
model_ckpt = "Dundalia/TWOLAR-xl"
score_strategy = "difference"
get_score = getattr(Score, score_strategy)
```
Than import the model and the tokenizer using the HF framework:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
```
Let's define a simple example: 
```python
query = "What is the significance of the Turing Award in the field of computer science?"

documents = [
    "Title: Major Milestones in Computer Science. Content: Another significant recognition in the world of computing is the Turing Award. Named after Alan Turing, the father of theoretical computer science and artificial intelligence, the award is often dubbed the 'Nobel Prize of Computer Science.' Established in 1966, it is conferred annually by the Association for Computing Machinery (ACM) to an individual or a group who made major contributions to the field of computer science.",
    "Title: Alan Turing, a Brief Biography. Content: Despite his monumental contributions, Turing faced persecution in his lifetime due to his sexuality. However, his legacy lives on in various ways, one of which is the Turing Award. While this award bears his name and is given in the domain of computer science, its significance in terms of academic recognition remains underappreciated.",
    "Title: Notbale Awards in Various Disciplines. Content: Turning to the realm of computer science, one cannot forget the Turing Award. The Turing Award has been around for decades and is a prestigious accolade, honoring those who have pushed the boundaries of computer science research and innovation. This award is more than just a title; it carries with it a rich history and a commitment to recognizing excellence in computing.",
]
```
Now we have to create the inputs for the model with the monoT5 prompt template and obtain the features to feed the model:
```python
inputs = [
    f"Query: {query} Document: {document} Relevant: " 
    for document in documents
]

features = tokenizer(
    inputs, truncation = True, 
    return_tensors = "pt", max_length = 500, 
    padding = True,
)

input_ids = features.input_ids
attention_mask = features.attention_mask
decode_ids = torch.full((input_ids.size(0), 1),
                         model.config.decoder_start_token_id,
                         dtype=torch.long)
```
Finally we can call the model and obtain the relevance scores for each document:
```python
with torch.no_grad():
    output = model(
            input_ids = input_ids.to(device), 
            attention_mask = attention_mask.to(device), 
            decoder_input_ids = decode_ids.to(device),    
        ) 
    logits = output.logits[:,0,:]
scores = get_score(logits).cpu().tolist()

for k in range(3):
    print(f"Document {k+1} Score: {round(scores[k], 4)}")
```
Expected output:
```
Document 1 Score: 2.6042
Document 2 Score: 1.3059
Document 3 Score: 1.6411
```

# Training

To train your own models using the provided scripts, use the `train.py` script as follows:

```bash
python3 train.py \
--base_model google/flan-t5-xl \
--train_path $YOUR_TRAIN_PATH \
--test_path $YOUR_TEST_PATH \
--outdir $YOUR_OUTDIR \
--epochs 1 \
--gradient_accumulation_steps 32 \
--loss_type rank_net \
--score_type difference
```

# Evaluation

Our evaluation scripts work for our models `TWOLAR-xl`, `TWOLAR-large`, but also for the monoT5 checkpoints available on the HuggingFace Hub, with `castorini/monot5-large-msmarco`, `castorini/monot5-3b-msmarco`, `castorini/monot5-large-msmarco-10k`, `castorini/monot5-3b-msmarco-10k` among others. In this case the `score_strategy` arg must be set to `softmax`.

Our evaluation scripts generate run files in the format described in the [next section](#run-files-format). 
For evaluation, we follow the [instruction to build](https://github.com/castorini/anserini-tools/tree/95fbaf2af75e2b59304ac5702d5479d50f3bd9ef) `trec_eval.9.0.4` so that we do not have to install any package for evaluation.

Once built, we can compute the nDCG@1, nDCG@5, and nDCG@10 as follows:

```bash
./anserini-tools/eval/trec_eval.9.0.4/trec_eval -c -m ndcg_cut.1,5,10 $YOUR_QRELS_FILE_PATH $YOUR_RESULTS_FILE_PATH
```

### Run files format

The result of each retrieval or reranking process is organized into folders named after the retrieval method followed by the specific corpus from the BEIR benchmark. For instance, you would have a folder structure like `results/bm25/nfcorpus`.
Each of these files is a `.txt` file containing the retrieval or reranking results. Each row in these files adheres to the BEIR conventional format:
```bash
qid Q0 pid rank score run_id
```
Here's a breakdown of what each field represents:
- `qid`: Query ID
- `Q0`: A static placeholder (it's a convention in TREC-style results)
- `pid`: Passage or Document ID
- `rank`: Rank of the document for the given query
- `score`: Retrieval score
- `run_id`: ID of the specific run or experiment


### Retrieval

For the retrieval phase we use BM25, DRAGON, and SPLADE models. We strictly adhere to the methodologies laid out in their respective repositories. If you're unfamiliar with their processes or need specific details on how they're implemented, please refer to their original repositories:
- BM25: [pyserini](https://github.com/castorini/pyserini/)
- DRAGON: [dpr-scale/dragon](https://github.com/facebookresearch/dpr-scale/tree/main/dragon)
- SPLADE: [splade](https://github.com/naver/splade/tree/main)

## Evaluating on TREC-DL2019 and TREC-DL2020

For the evaluation on the MSMARCO dataset we have adopted test sets of the 2019 and 2020 competitions: TREC-DL2019 and TREC-DL2020. 
In the `eval_trec_dl.py` script, we directly retrieve the top 100 documents for each query using the pyserini API. To that aim we indexed and stored the indexes following the [instructions](https://github.com/castorini/pyserini/blob/master/docs/usage-index.md#building-a-bm25-index-embeddable-python-implementation). 

We have downloaded the queries and qrels from their [github](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2020.html).

```bash
mkdir $YOUR_TRECDL_FOLDER
cd $YOUR_TRECDL_FOLDER

wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz
gunzip msmarco-test2019-queries.tsv.gz
wget https://trec.nist.gov/data/deep/2019qrels-pass.txt

wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz
gunzip msmarco-test2020-queries.tsv.gz
wget https://trec.nist.gov/data/deep/2020qrels-pass.txt

cd ..
```

The `eval_trec_dl.py` script assumes that the files `msmarco-test2019-queries.tsv`, `2019qrels-pass.txt`, `msmarco-test2020-queries.tsv`, `2020qrels-pass.txt` are stored into the `$YOUR_TRECDL_FOLDER` folder. It will create two files in the aforementioned run file format: `$YOUR_OUTDIR/trec-dl-2019` and `$YOUR_OUTDIR/trec-dl-2020`.

```bash
python3 eval_trec_dl.py \
--trecdl_path $YOUR_TRECDL_FOLDER \
--index_path $YOUR_INDEX_PATH \
--corpus_path $YOUR_CORPUS_PATH \
--model_ckpt $YOUR_MODEL_CKPT_PATH \
--outdir $YOUR_OUTDIR \
--score_strategy "difference" \
--topk 100 \
```

## Evaluating on BEIR

### Download

We adopted the preprocessing of the [DRAGON repository](https://github.com/facebookresearch/dpr-scale/tree/main/dragon#dragon_beir_eval). Here we use the nfcorpus dataset as an example.
We first have to download BEIR datasets from its [github](https://github.com/beir-cellar/beir).

Following the DRAGON implementation, we first download and decompress the dataset; then, using their script we preprocess the corpus and queries. We also transform the `qrels.test.tsv` into trec qrel fromat. 

```bash
# data download and decompress
mkdir $YOUR_BEIR_FOLDER
cd $YOUR_BEIR_FOLDER
wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip
unzip nfcorpus.zip
cd ..

# data preprocess
python3 prep_beir_eval.py --data_dir $YOUR_BEIR_FOLDER/nfcorpus
```

This will create three files `collection.tsv`, `queries.test.tsv`, and `qrels.test.tsv` formatted respectively as follows:

```bash
- collection.tsv
pid \t document

- queries.test.tsv
qid \t query

- qrels.test.tsv
qid 0 pid score
```

### Reranking

At this point we can utilize the `eval_beir.py` to rerank the documents. The script assumes that under the `$YOUR_PATH_TO_RUNFILE` folder are contained the run files of the results of the first-stage retriever for each of the BEIR datasets. 

```bash
python3 eval_beir.py \
--path_to_runfile $YOUR_PATH_TO_RUNFILE \
--beir_folder $YOUR_BEIR_FOLDER \
--model_ckpt $YOUR_MODEL_CHECKPOINT_PATH \
--score_strategy difference \
--n_docs 100 \
```







