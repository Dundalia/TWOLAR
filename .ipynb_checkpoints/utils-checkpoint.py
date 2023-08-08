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
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.nn import BCELoss, BCEWithLogitsLoss, CrossEntropyLoss
from itertools import product

#################################################
#########   RankLoss          ###################
#################################################

class RankLoss:

    @staticmethod
    def softmax_ce_loss(y_pred, *args, **kwargs):
        return F.cross_entropy(y_pred, torch.zeros((y_pred.size(0),)).long().cuda())

    @staticmethod
    def pointwise_rmse(y_pred, y_true=None):
        if y_true is None:
            y_true = torch.zeros_like(y_pred).to(y_pred.device)
            y_true[:, 0] = 1
        errors = (y_true - y_pred)
        squared_errors = errors ** 2
        valid_mask = (y_true != -100).float()
        mean_squared_errors = torch.sum(squared_errors, dim=1) / torch.sum(valid_mask, dim=1)
        rmses = torch.sqrt(mean_squared_errors)
        return torch.mean(rmses)

    @staticmethod
    def pointwise_bce(y_pred, y_true=None):
        if y_true is None:
            y_true = torch.zeros_like(y_pred).float().to(y_pred.device)
            y_true[:, 0] = 1
        loss = F.binary_cross_entropy(torch.sigmoid(y_pred), y_true)
        return loss
    
    @staticmethod
    def rank_net(y_pred, y_true=None):
        """
        RankNet loss introduced in "Learning to Rank using Gradient Descent".
        :param y_pred: predictions from the model, shape [batch_size, slate_length]
        :param y_true: ground truth labels, shape [batch_size, slate_length]
        """
        if y_true is None:
            y_true = torch.zeros_like(y_pred).to(y_pred.device)
            y_true[:, 0] = 1

        # here we generate every pair of indices from the range of document length in the batch
        document_pairs_candidates = list(product(range(y_true.shape[1]), repeat=2))

        pairs_true = y_true[:, document_pairs_candidates]
        true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]
        the_mask = (true_diffs > 0) & (~torch.isinf(true_diffs)).view(-1)
        selected_pred = y_pred[:, document_pairs_candidates][the_mask].view(-1,2)

        y_true = torch.zeros_like(selected_pred, dtype=torch.int64)[:,0]

        return CrossEntropyLoss()(selected_pred, y_true)

    @staticmethod
    def list_net(y_pred, y_true=None, padded_value_indicator=-100, eps=1e-10):
        """
            ListNet loss introduced in "Learning to Rank: From Pairwise Approach to Listwise Approach".
            :param y_pred: predictions from the model, shape [batch_size, slate_length]
            :param y_true: ground truth labels, shape [batch_size, slate_length]
            :param eps: epsilon value, used for numerical stability
            :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
            :return: loss value, a torch.Tensor
            """
        if y_true is None:
            y_true = torch.zeros_like(y_pred).to(y_pred.device)
            y_true[:, 0] = 1

        preds_smax = F.softmax(y_pred, dim=1)
        true_smax = F.softmax(y_true, dim=1)

        preds_smax = preds_smax + eps
        preds_log = torch.log(preds_smax)

        return torch.mean(-torch.sum(true_smax * preds_log, dim=1))

    @staticmethod
    def lambda_loss(y_pred, y_true=None, eps=1e-10, padded_value_indicator=-100, weighing_scheme=None, k=None,
                    sigma=1., mu=10., reduction="mean", reduction_log="binary"):
        """
        LambdaLoss framework for LTR losses implementations, introduced in "The LambdaLoss Framework for Ranking Metric Optimization".
        Contains implementations of different weighing schemes corresponding to e.g. LambdaRank or RankNet.
        :param y_pred: predictions from the model, shape [batch_size, slate_length]
        :param y_true: ground truth labels, shape [batch_size, slate_length]
        :param eps: epsilon value, used for numerical stability
        :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
        :param weighing_scheme: a string corresponding to a name of one of the weighing schemes
        :param k: rank at which the loss is truncated
        :param sigma: score difference weight used in the sigmoid function
        :param mu: optional weight used in NDCGLoss2++ weighing scheme
        :param reduction: losses reduction method, could be either a sum or a mean
        :param reduction_log: logarithm variant used prior to masking and loss reduction, either binary or natural
        :return: loss value, a torch.Tensor
        """
        if y_true is None:
            y_true = torch.zeros_like(y_pred).to(y_pred.device)
            y_true[:, 0] = 1

        device = y_pred.device

        # Here we sort the true and predicted relevancy scores.
        y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
        y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

        # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
        true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
        true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
        padded_pairs_mask = torch.isfinite(true_diffs)

        if weighing_scheme != "ndcgLoss1_scheme":
            padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)

        ndcg_at_k_mask = torch.zeros((y_pred.shape[1], y_pred.shape[1]), dtype=torch.bool, device=device)
        ndcg_at_k_mask[:k, :k] = 1

        # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
        true_sorted_by_preds.clamp_(min=0.)
        y_true_sorted.clamp_(min=0.)

        # Here we find the gains, discounts and ideal DCGs per slate.
        pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
        D = torch.log2(1. + pos_idxs.float())[None, :]
        maxDCGs = torch.sum(((torch.pow(2, y_true_sorted) - 1) / D)[:, :k], dim=-1).clamp(min=eps)
        G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

        # Here we apply appropriate weighing scheme - ndcgLoss1, ndcgLoss2, ndcgLoss2++ or no weights (=1.0)
        if weighing_scheme is None:
            weights = 1.
        else:
            weights = globals()[weighing_scheme](G, D, mu, true_sorted_by_preds)  # type: ignore

        # We are clamping the array entries to maintain correct backprop (log(0) and division by 0)
        scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-1e8, max=1e8)
        scores_diffs.masked_fill(torch.isnan(scores_diffs), 0.)
        weighted_probas = (torch.sigmoid(sigma * scores_diffs).clamp(min=eps) ** weights).clamp(min=eps)
        if reduction_log == "natural":
            losses = torch.log(weighted_probas)
        elif reduction_log == "binary":
            losses = torch.log2(weighted_probas)
        else:
            raise ValueError("Reduction logarithm base can be either natural or binary")

        if reduction == "sum":
            loss = -torch.sum(losses[padded_pairs_mask & ndcg_at_k_mask])
        elif reduction == "mean":
            loss = -torch.mean(losses[padded_pairs_mask & ndcg_at_k_mask])
        else:
            raise ValueError("Reduction method can be either sum or mean")

        return loss


def ndcgLoss1_scheme(G, D, *args):
    return (G / D)[:, :, None]


def ndcgLoss2_scheme(G, D, *args):
    pos_idxs = torch.arange(1, G.shape[1] + 1, device=G.device)
    delta_idxs = torch.abs(pos_idxs[:, None] - pos_idxs[None, :])
    deltas = torch.abs(torch.pow(torch.abs(D[0, delta_idxs - 1]), -1.) - torch.pow(torch.abs(D[0, delta_idxs]), -1.))
    deltas.diagonal().zero_()

    return deltas[None, :, :] * torch.abs(G[:, :, None] - G[:, None, :])


def lambdaRank_scheme(G, D, *args):
    return torch.abs(torch.pow(D[:, :, None], -1.) - torch.pow(D[:, None, :], -1.)) * torch.abs(
        G[:, :, None] - G[:, None, :])


def ndcgLoss2PP_scheme(G, D, *args):
    return args[0] * ndcgLoss2_scheme(G, D) + lambdaRank_scheme(G, D)


def rankNet_scheme(G, D, *args):
    return 1.


def rankNetWeightedByGTDiff_scheme(G, D, *args):
    return torch.abs(args[1][:, :, None] - args[1][:, None, :])


def rankNetWeightedByGTDiffPowed_scheme(G, D, *args):
    return torch.abs(torch.pow(args[1][:, :, None], 2) - torch.pow(args[1][:, None, :], 2))
    

#################################################
#########   get score         ###################
#################################################

class Score:
    
    @staticmethod
    def extra_id(logits):
        return logits[:, 32089]
    
    @staticmethod
    def difference(logits):
        true_logits = logits[:, 1176]
        false_logits = logits[:, 6136]
        return true_logits - false_logits
    
    @staticmethod
    def softmax(logits):
        true_logits = logits[:, 1176]
        false_logits = logits[:, 6136]
        scores = [torch.nn.functional.softmax(torch.stack([true_logit, false_logit]), dim=0)[0]
                 for true_logit, false_logit in zip(true_logits, false_logits)]
        return torch.stack(scores) 
        

#################################################
#########   Rerank Data       ###################
#################################################

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

##########

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


def read_beir(beir_folder, corpus):
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
    
