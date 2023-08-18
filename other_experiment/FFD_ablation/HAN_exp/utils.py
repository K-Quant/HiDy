import datetime
import errno
import os
import pickle
import random
from pprint import pprint

import dgl
import pandas as pd
import numpy as np
import torch
from dgl.data.utils import _get_dgl_url, download, get_download_dir
from scipy import io as sio, sparse
import pickle

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def mkdir_p(path, log=True):
    try:
        os.makedirs(path)
        if log:
            print("Created directory {}".format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print("Directory {} already exists.".format(path))
        else:
            raise


def get_date_postfix():
    dt = datetime.datetime.now()
    post_fix = "{}_{:02d}-{:02d}-{:02d}".format(
        dt.date(), dt.hour, dt.minute, dt.second
    )

    return post_fix




default_configure = {
    "lr": 0.001,  # Learning rate
    "num_heads": [8],  # Number of attention heads for node-level attention
    "hidden_units": 8,
    "dropout": 0.6,
    "weight_decay": 0.001,
    "num_epochs": 200,
    "patience": 20,
}


sampling_configure = {"batch_size": 50}


def setup(args):
    args.update(default_configure)
    set_random_seed(args["seed"])
    return args


def setup_for_sampling(args):
    args.update(default_configure)
    args.update(sampling_configure)
    set_random_seed()
    args["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    return args


def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()




def load_data(args, remove_self_loop=False):
    filename = 'sparse_matrices.pkl'
    with open(filename, 'rb') as file:
        data = pickle.load(file)

    matrix_lst = ['adj_CPC', 'adj_CIC', 'adj_CFC', 'adj_CAC', 'dispute', 'same_industry', 'fall', 'cooperate',
                  'invest', 'rise', 'increase_holding', 'superior', 'supply', 'be_increased_holding', 'be_reduced_holding',
                  'be_invested', 'reduce_holding']
    branch_dic = {'IndustryChain': ['adj_CPC', 'supply'],
                  'SectorIndustry': ['adj_CIC', 'same_industry','superior'],
                  'Ownership':['invest','increase_holding','be_increased_holding','be_reduced_holding','be_invested','reduce_holding'],
                  'Partnership': ['dispute','fall','cooperate','rise']}
    csmar = ['adj_CFC', 'adj_CAC']
    num_classes = 1


    num_nodes = data["label"].shape[0]
    if remove_self_loop:
        for i in matrix_lst:
            data[i] = data[i] - np.eye(num_nodes)  # 减去单位矩->去除self-loop

    gs = []
    print("args['Branch_lst']", args['Branch_lst'])
    if args['Branch_lst'] != None:
        for i in args['Branch_lst']:
            for j in branch_dic[i]:
                gs.append(dgl.from_scipy(data[j]))
    for i in csmar:
        gs.append(dgl.from_scipy(data[i]))

    train_mask = torch.tensor(data['train_idx'], dtype=torch.long)
    val_mask = torch.tensor(data['val_idx'], dtype=torch.long)
    test_mask = torch.tensor(data['test_idx'], dtype=torch.long)


    # loading feature
    if args['Branch_lst'] == None:
        embedding_path = '../features/CSMAR.csv'
    elif len(args['Branch_lst']) == 1:
        if args['Branch_lst'] == ['IndustryChain']:
            embedding_path = '../features/CSMAR+IndustryChain.csv'
        elif args['Branch_lst'] == ['SectorIndustry']:
            embedding_path = '../features/CSMAR+SectorIndustry.csv'
        elif args['Branch_lst'] == ['Ownership']:
            embedding_path = '../features/CSMAR+Ownership.csv'
        elif args['Branch_lst'] == ['Partnership']:
            embedding_path = '../features/CSMAR+Partnership.csv'
        else:
            print("ERROR Branch_lst")
            exit()
    elif len(args['Branch_lst']) == 2:
        if 'IndustryChain' in args['Branch_lst'] and 'SectorIndustry' in args['Branch_lst']:
            embedding_path = '../features/CSMAR+IndustryChain+SectorIndustry.csv'
        elif 'IndustryChain' in args['Branch_lst'] and 'Ownership' in args['Branch_lst']:
            embedding_path = '../features/CSMAR+IndustryChain+Ownership.csv'
        elif 'IndustryChain' in args['Branch_lst'] and 'Partnership' in args['Branch_lst']:
            embedding_path = '../features/CSMAR+IndustryChain+Partnership.csv'
        elif 'Ownership' in args['Branch_lst'] and 'SectorIndustry' in args['Branch_lst']:
            embedding_path = '../features/CSMAR+Ownership+SectorIndustry.csv'
        elif 'Partnership' in args['Branch_lst'] and 'SectorIndustry' in args['Branch_lst']:
            embedding_path = '../features/CSMAR+Partnership+SectorIndustry.csv'
        elif 'Partnership' in args['Branch_lst'] and 'Ownership' in args['Branch_lst']:
            embedding_path = '../features/CSMAR+Ownership+Partnership.csv'
        else:
            print("ERROR Branch_lst")
            exit()
    elif len(args['Branch_lst']) == 3:
        if 'IndustryChain' in args['Branch_lst'] and 'SectorIndustry' in args['Branch_lst'] and 'Ownership' in args['Branch_lst']:
            embedding_path = '../features/CSMAR+Ownership+SectorIndustry+IndustryChain.csv'
        elif 'IndustryChain' in args['Branch_lst'] and 'SectorIndustry' in args['Branch_lst'] and 'Partnership' in args['Branch_lst']:
            embedding_path = '../features/CSMAR+Partnership+SectorIndustry+IndustryChain.csv'
        elif 'Ownership' in args['Branch_lst'] and 'SectorIndustry' in args['Branch_lst'] and 'Partnership' in args['Branch_lst']:
            embedding_path = '../features/CSMAR+Partnership+SectorIndustry+Ownership.csv'
        elif 'Ownership' in args['Branch_lst'] and 'IndustryChain' in args['Branch_lst'] and 'Partnership' in args['Branch_lst']:
            embedding_path = '../features/CSMAR+Partnership+IndustryChain+Ownership.csv'
        else:
            print("ERROR Branch_lst")
            exit()
    elif len(args['Branch_lst']) == 4:
        embedding_path = '../features/CSMAR+Partnership+IndustryChain+Ownership+SectorIndustry.csv'
    else:
        print("ERROR Branch_lst")
        exit()

    features = pd.read_csv(embedding_path)
    dataset_X = list()
    for idx in range(len(features)):
        embedding_list = features["embeddings"][idx][1:][:-1].split(",")
        dataset_X.append(np.array(embedding_list))
    dataset_X = np.array(dataset_X).astype(np.float)
    dataset_X = torch.tensor(dataset_X)

    return (
        gs,
        dataset_X,
        data["label"],
        num_classes,
        data['train_idx'],
        data['val_idx'],
        data['test_idx'],
        train_mask,
        val_mask,
        test_mask,
    )



class EarlyStopping(object):
    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        self.filename = "early_stop_{}_{:02d}-{:02d}-{:02d}.pth".format(
            dt.date(), dt.hour, dt.minute, dt.second
        )
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            # print(
            #     f"EarlyStopping counter: {self.counter} out of {self.patience}"
            # )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))