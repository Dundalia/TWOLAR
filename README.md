# TWOLAR: a TWO-steps LLM-Augmented distillation method for passage Reranking

This repository provides the implementation of the paper "[TWOLAR: a TWO-steps LLM-Augmented distillation method for passage Reranking](google.com)". The TWOLAR method introduces a novel approach to passage reranking, leveraging two-step distillation in combination with language model augmentation to improve results in various reranking benchmarks.

![data-generation](/images/data-generation.png)
*Illustration of the methodology adopted to build the distillation dataset.*

## Table of Contents
- [A quick example](#a-quick-example)
- [Training](#training)
- [Evaluation](#evaluation)
  - [Run files format](#run-files-format)   
  - [Evaluating on BEIR](#evaluating-on-beir)
    - [Download](#download)
    - [Retrieval](#retrieval)
    - [Reranking](#reranking)
    - [Evaluation](#evaluation)

# A quick example
Here we explain how to utilize the models. The code for this example can be found in the notebook `example.ipynb`. 
First of all, define the model checkpoint and the corresponding score strategy:
```python
from utils import Score
model_ckpt = "..."
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

## Evaluating on BEIR

### Download

We adopted the preprocessing of the [DRAGON repository](https://github.com/facebookresearch/dpr-scale/tree/main/dragon#dragon_beir_eval). Here we use the nfcorpus dataset as an example.
We first have to download BEIR datasets from its [github](https://github.com/beir-cellar/beir).

Following the DRAGON implementation, we first download decompress the dataset; then, using their script we preprocess the corpus and queries. We also transform the `qrels.test.tsv` into trec qrel fromat. 

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


### Reranking

At this point we can utilize the `rerank.py` script as follows:

```bash
python3 rerank.py \
--corpus nfcorpus \
--path_to_runfile $YOUR_PATH_TO_RUNFILE \
--beir_folder $YOUR_BEIR_FOLDER \
--model_ckpt $YOUR_MODEL_CHECKPOINT_PATH \
--score_strategy difference \
--n_docs 100 \
```

### Evaluation

For evaluation, we folow the [instruction to build](https://github.com/castorini/anserini-tools/tree/95fbaf2af75e2b59304ac5702d5479d50f3bd9ef) `trec_eval.9.0.4` so that we do not have to install any package for evaluation.





