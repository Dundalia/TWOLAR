import os
import json
import pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset
import argparse
from rank_loss import RankLoss
from accelerate import Accelerator
from sklearn.metrics import ndcg_score
import lightning as L
from lightning.fabric.strategies import FSDPStrategy
from copy import deepcopy

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    AdamW
)
from jsonargparse import CLI

score_type = "difference"
loss_type = "pairwise_ce"
eval_interval = 500
neg_num = 29
gradient_accumulation_steps = 32
micro_batch_size = 10
in_mb = False


class RerankData(Dataset):
    def __init__(self, data, tokenizer, neg_num=20, label=True):
        self.data = data
        self.tokenizer = tokenizer
        self.neg_num = neg_num
        self.label = label
        if not label:
            self.neg_num += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        item = self.data[item]
        query = item['query']

        if self.label:
            pos = [str(item['positive_passages'][0]['text'])]
            pos_id = [psg['docid'] for psg in item['positive_passages']]
            neg = [str(psg['text']) for psg in item['retrieved_passages'] if psg['docid'] not in pos_id][:self.neg_num]
        else:
            pos = []
            neg = [str(psg['text']) for psg in item['retrieved_passages']][:self.neg_num]
        neg = neg + ['<padding_passage>'] * (self.neg_num - len(neg))
        passages = pos + neg
        return query, passages

    def collate_fn(self, data):
        query, passages = list(*data)
        inputs = [f"Query: {query} Document: {passage}" for passage in passages]
        
        features = self.tokenizer(
            inputs, 
            truncation = True, return_tensors="pt", 
            max_length=500, padding=True
        )
    
        return features

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
        scores = [torch.nn.functional.softmax(torch.stack([true_logit, false_logit]), dim=0)[0]
                 for true_logit, false_logit in zip(true_logits, false_logits)]
        return torch.stack(scores) 


def main(
    base_model: str = "google/flan-t5-large", 
    data_path: str = "/data/davide/rankgpt-data/total-gpt3.5-train.jsonl",
    test_path: str = "/data/davide/rankgpt-data/total-gpt3.5-test.jsonl",
    out_dir: str = "/data/davide/models/t5/rankt5/flan-t5-large-difference-total-30-ga-32",
    epochs: int = 3,
    save_at_epoch_end: bool = True,
):

    accelerator = Accelerator(gradient_accumulation_steps = gradient_accumulation_steps)
    #accelerator = Accelerator()
    
    ## Prepare model and optimizer
    conf = AutoConfig.from_pretrained(base_model)
    conf.use_cache = False
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model, config=conf)
    model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5) ## 5e-5
    
    ## Prepare data
    train_data = [json.loads(line) for line in open(data_path)]
    test_data = [json.loads(line) for line in open(test_path)]
    train_dataset = RerankData(train_data, tokenizer, neg_num=neg_num, label=False)
    test_dataset = RerankData(test_data, tokenizer, neg_num=29, label=False) ## testing with 30 docs always
    train_dl = torch.utils.data.DataLoader(
        train_dataset, collate_fn=train_dataset.collate_fn,
        batch_size=1, shuffle=True, num_workers=0)
    test_dl = torch.utils.data.DataLoader(
        test_dataset, collate_fn=test_dataset.collate_fn,
        batch_size=1, shuffle=True, num_workers=0)

    ## loss function
    loss_function = getattr(RankLoss, loss_type)

    model, optimizer, train_dl, test_dl = accelerator.prepare(
        model, optimizer, train_dl, test_dl
    )
    
    # Train
    for epoch in range(epochs):
        accelerator.print(f'Training {out_dir} epoch {epoch+1} of {epochs}')
        model.train()
        tk0 = tqdm(enumerate(train_dl), total=len(train_dl), leave=False)
        loss_report = []
        for step_count, batch in tk0:
            input_ids = batch.input_ids
            attention_mask = batch.attention_mask            
            decode_ids = torch.full((input_ids.size(0), 1),
                             model.config.decoder_start_token_id,
                             dtype=torch.long)

            ## In micro batches
            if in_mb:
                #with accelerator.accumulate(model):
                    scores = []
                    for k in range(input_ids.shape[0] // micro_batch_size):
                        print(k)
                        start_idx = k * micro_batch_size
                        end_idx = start_idx + micro_batch_size
                        mask = torch.zeros(input_ids.size(0), dtype=torch.bool)
                        mask[start_idx:end_idx] = True
                            
                        micro_input_ids = input_ids[mask]
                        micro_attention_mask = attention_mask[mask]
                        micro_decode_ids = decode_ids[mask]
                        
                        micro_output = model(
                            input_ids = micro_input_ids, 
                            attention_mask = micro_attention_mask, 
                            decoder_input_ids = micro_decode_ids.to(accelerator.device),    
                        )
        
                        micro_logits = micro_output.logits[:,0,:]
                        scores.append(get_score(micro_logits))
        
                    scores = torch.cat(scores).view(1,-1)
                    y_true = torch.tensor([[1 / (i + 1) 
                                            for i in range(scores.size(1))]] *
                                            scores.size(0))
                    
                    loss = loss_function(scores, y_true.type_as(scores))
                    loss_report.append(loss.item())
                    ## Backward pass
                    accelerator.backward(loss)
                    accelerator.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    optimizer.zero_grad()
                

            # pass all the num_neg inputs together 
            else:
                with accelerator.accumulate(model):
                    output = model(
                        input_ids = input_ids, 
                        attention_mask = attention_mask, 
                        decoder_input_ids = decode_ids.to(accelerator.device),    
                    )
                    logits = output.logits[:,0,:]
                    scores = get_score(logits).view(1,-1)

                    ## Untab from here to optimmizer.zero_grad()
                    y_true = torch.tensor([[1 / (i + 1) 
                                            for i in range(scores.size(1))]] *
                                            scores.size(0))
                    
                    loss = loss_function(scores, y_true.type_as(scores))
                    loss_report.append(loss.item())
                    ## Backward pass
                    accelerator.backward(loss)
                    accelerator.clip_grad_norm_(model.parameters(), 1.)
                    
                    #params_before = [p.clone().detach() for p in list(model.parameters())[:2] + list(model.parameters())[-2:]]
                    
                    optimizer.step()

                    ## Check if parameters have been updated
                    #params_after = [p.clone().detach() for p in list(model.parameters())[:2] + list(model.parameters())[-2:]]
                    #for p_before, p_after in zip(params_before, params_after):
                    #    if not torch.all(torch.eq(p_before, p_after)):
                    #        print("Parameters updated - ", step_count)
                    #        break

                    optimizer.zero_grad()

            ## Validate
            if (step_count % eval_interval == 0) and step_count:
                model.eval()
                tk1 = tqdm(test_dl, total=len(test_dl), leave=False)
                test_loss_report = []
                test_ndcg = []
                with torch.no_grad():
                    for test_batch in tk1:
                        input_ids = test_batch.input_ids
                        attention_mask = test_batch.attention_mask
                        decode_ids = torch.full((input_ids.size(0), 1),
                                         model.config.decoder_start_token_id,
                                         dtype=torch.long)
                        output = model(
                            input_ids = input_ids, 
                            attention_mask = attention_mask, 
                            decoder_input_ids = decode_ids.to(accelerator.device),    
                        ) 
                        logits = output.logits[:,0,:]
                        scores = get_score(logits).view(1,-1)
                        y_true = torch.tensor([[1 / (i + 1) 
                                                for i in range(scores.size(1))]] *
                                              scores.size(0))
                        loss = loss_function(scores, y_true.to(accelerator.device))
                        test_loss_report.append(accelerator.gather(loss).mean().item())
                        test_ndcg.append(ndcg_score(y_true.cpu(), scores.cpu()))
                model.train()
                accelerator.print(f"step: {step_count} val loss: {np.mean(test_loss_report):.4f} val ndcg: {np.mean(test_ndcg):.4f}")    
                
            tk0.set_postfix(loss=sum(loss_report[-100:]) / len(loss_report[-100:]))
        if save_at_epoch_end or epoch==epochs-1:
            ckpt_path = os.path.join(out_dir, f"checkpoint-{epoch+1}")
            os.makedirs(ckpt_path, exist_ok=True)
            unwrap_model = accelerator.unwrap_model(model)
            unwrap_model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)


    

    

    
    


















    


if __name__ == "__main__":
    CLI(main)