import os
import json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset
from accelerate import Accelerator
from sklearn.metrics import ndcg_score
from jsonargparse import CLI
from utils import RankLoss, RerankData, Score
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
)


def main(
    base_model: str, 
    train_path: str,
    test_path: str,
    outdir: str,
    epochs: int = 1,
    eval_interval: int = 500, 
    neg_num: int = 29, 
    gradient_accumulation_steps: int = 32, 
    save_at_epoch_end: bool = True,
    loss_type: str = "rank_net", 
    score_strategy: str = "difference",
):

    # Initializing accelerator for gradient accumulation
    accelerator = Accelerator(gradient_accumulation_steps = gradient_accumulation_steps)
    
    # Load model, its conf, and tokenizer
    conf = AutoConfig.from_pretrained(base_model)
    conf.use_cache = False
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model, config=conf)
    model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5) 
    
    # Loading training and testing data
    train_data = [json.loads(line) for line in open(train_path)]
    test_data = [json.loads(line) for line in open(test_path)]

    # Create datasets for training and testing
    train_dataset = RerankData(train_data, tokenizer, neg_num=neg_num, label=False)
    test_dataset = RerankData(test_data, tokenizer, neg_num=neg_num, label=False) 

    # Define dataloaders for training and testing
    train_dl = torch.utils.data.DataLoader(
        train_dataset, collate_fn=train_dataset.collate_fn,
        batch_size=1, shuffle=True, num_workers=0)
    test_dl = torch.utils.data.DataLoader(
        test_dataset, collate_fn=test_dataset.collate_fn,
        batch_size=1, shuffle=True, num_workers=0)

    # Set up loss function and scoring strategy
    loss_function = getattr(RankLoss, loss_type)
    get_score = getattr(Score, score_strategy)
    
    model, optimizer, train_dl, test_dl = accelerator.prepare(
        model, optimizer, train_dl, test_dl
    )
    
    # Training loop
    for epoch in range(epochs):
        accelerator.print(f'Training {outdir} epoch {epoch+1} of {epochs}')
        model.train()
        tk0 = tqdm(enumerate(train_dl), total=len(train_dl), leave=False)
        loss_report = []
        for step_count, batch in tk0:
            input_ids = batch.input_ids
            attention_mask = batch.attention_mask
            # Initialize decode ids for model input
            decode_ids = torch.full((input_ids.size(0), 1),
                             model.config.decoder_start_token_id,
                             dtype=torch.long)

            with accelerator.accumulate(model):
                # Forward pass
                output = model(
                    input_ids = input_ids, 
                    attention_mask = attention_mask, 
                    decoder_input_ids = decode_ids.to(accelerator.device),    
                )
                logits = output.logits[:,0,:]
                scores = get_score(logits).view(1,-1)

                # Compute scores of the ground truth (GPT) order  
                y_true = torch.tensor([[1 / (i + 1) 
                                        for i in range(scores.size(1))]] *
                                        scores.size(0))

                # Calculate loss
                loss = loss_function(scores, y_true.type_as(scores))
                loss_report.append(loss.item())
                
                # Backward pass and optimizer step
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                optimizer.zero_grad()

            # Validate at every 'eval_interval'
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
                    tk1.close()
                model.train()
                # Print validation metrics
                tqdm.write(f"iter: {step_count} step: {step_count // gradient_accumulation_steps} val loss: {np.mean(test_loss_report):.4f} val ndcg: {np.mean(test_ndcg):.4f}")   

            # Update training loss on progress bar
            tk0.set_postfix(loss=sum(loss_report[-100:]) / len(loss_report[-100:]))

        # Save model at epoch end or at end of training
        if save_at_epoch_end or epoch==epochs-1:
            ckpt_path = os.path.join(outdir, f"checkpoint-{epoch+1}")
            os.makedirs(ckpt_path, exist_ok=True)
            unwrap_model = accelerator.unwrap_model(model)
            unwrap_model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)


if __name__ == "__main__":
    CLI(main)
