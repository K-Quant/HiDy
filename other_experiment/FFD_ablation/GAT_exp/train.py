from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import f1_score
import pandas as pd
from utils import load_data, accuracy
from models import GAT, SpGAT


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--Branch_lst', default=['IndustryChain'], help='Patience')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()



def score(logits, labels):
    prediction = [0 if x < 0.5 else 1 for x in logits]
    labels = labels.cpu().numpy()
    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average="micro")
    macro_f1 = f1_score(labels, prediction, average="macro")
    return accuracy, micro_f1, macro_f1

def train(epoch):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    if args.cuda:
        output = output.cuda()

    loss = loss_fcn(output[idx_train].squeeze(), labels[idx_train].float().squeeze())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if not args.fastmode:
        model.eval()
        output = model(features, adj)
    loss_val = loss_fcn(output[idx_val].squeeze(), labels[idx_val].float().squeeze())
    val_acc, val_micro_f1, val_macro_f1 = score(
        output[idx_val].squeeze(), labels[idx_val].float().squeeze()
    )
    print(
        "Epoch {} | Valid accuracy {:.4f} | Valid Loss {:.4f} | Valid Micro f1 {:.4f} | Valid Macro f1 {:.4f} ".format(
            epoch,
            val_acc,
            loss_val.item(),
            val_micro_f1,
            val_macro_f1,
        )
    )
    return loss_val.data.item()


def compute_test():
    model.eval()
    output = model(features, adj)
    # loss_test = loss_fcn(output[idx_test].squeeze(), labels[idx_test].float().squeeze())
    test_acc, test_micro_f1, test_macro_f1 = score(output[idx_test].squeeze(), labels[idx_test].float().squeeze())
    print(
        "Test accuracy {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f} ".format(
            test_acc,
            test_micro_f1,
            test_macro_f1,
        )
    )
    return test_acc, test_micro_f1, test_macro_f1


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")



# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data(args.Branch_lst)

# Model and optimizer
if args.sparse:
    model = SpGAT(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=int(labels.max()) + 1,
                dropout=args.dropout,
                nheads=args.nb_heads,
                alpha=args.alpha).to(device)
else:
    model = GAT(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=1,
                dropout=args.dropout,
                nheads=args.nb_heads,
                alpha=args.alpha).to(device)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)

# sector industry is very sensitive, and therefore we need to assign different parameters
balance_weight = 3.2 if args.Branch_lst is None else (2.6 if 'SectorIndustry' in args.Branch_lst else 3.2)

features, adj, labels = Variable(features), Variable(adj), Variable(labels)
labels = labels.squeeze()

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


pos_weight = torch.tensor([balance_weight]).to(device)
loss_fcn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# Train model
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0

file_list = os.listdir('./ckpts/')
for file in file_list:
    file_path = os.path.join('./ckpts/', file)
    # print("remove", file_path)
    if os.path.isfile(file_path):  # 确保是文件而不是子文件夹
        os.remove(file_path)

# Training
for epoch in range(args.epochs):
    loss_values.append(train(epoch))

    torch.save(model.state_dict(), './ckpts/{}.pkl'.format(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1
    if bad_counter == args.patience:
        break

    files = glob.glob('./ckpts/*.pkl')
    for file in files:
        file_temp = file.split('\\')[-1]
        epoch_nb = int(file_temp.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob('./ckpts/*.pkl')
for file in files:
    file_temp = file.split('\\')[-1]
    epoch_nb = int(file_temp.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('./ckpts/{}.pkl'.format(best_epoch)))

# Testing
test_acc, test_micro_f1, test_macro_f1 = compute_test()

print("finished")