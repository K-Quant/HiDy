import torch
import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
from torchmetrics import RetrievalNormalizedDCG


def mse(pred, label):
    loss = (pred - label)**2
    return torch.mean(loss)


def mae(pred, label):
    loss = (pred - label).abs()
    return torch.mean(loss)


def metric_fn(preds):
    preds = preds[~np.isnan(preds['label'])]
    precision = {}
    recall = {}

    temp = preds.groupby(level='datetime').apply(lambda x: x.sort_values(by='score', ascending=False))
    if len(temp.index[0]) > 2:
        temp = temp.reset_index(level=0).drop('datetime', axis = 1)
        
    for k in [1, 3, 5, 10, 20, 30, 50, 100]:
        precision[k] = temp.groupby(level='datetime').apply(lambda x: (x.label[:k] > 0).sum()/k).mean()
        recall[k] = temp.groupby(level='datetime').apply(lambda x: (x.label[:k] > 0).sum()/(x.label > 0).sum()).mean()

    ic = preds.groupby(level='datetime').apply(lambda x: x.label.corr(x.score)).mean()
    rank_ic = preds.groupby(level='datetime').apply(lambda x: x.label.corr(x.score, method='spearman')).mean()
    return precision, recall, ic, rank_ic



