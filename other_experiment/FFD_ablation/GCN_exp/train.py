import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
import pandas as pd
import  numpy as np
from    data import load_data, preprocess_features, preprocess_adj
from    model import GCN
from    utils import masked_loss, masked_acc
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")


import  argparse

args = argparse.ArgumentParser()
args.add_argument('--model', default='gcn')
args.add_argument('--learning_rate', type=float, default=0.001)
args.add_argument('--epochs', type=int, default=200)
args.add_argument('--weight_decay', type=float, default=0.001)
args.add_argument('--seed', type=int, default=1)
args.add_argument('--Branch_lst', nargs="+", type=str, default = None, help="List of knowledge")
args.add_argument('--device', default='cuda')
args = args.parse_args()


def score(logits, labels):
    prediction = [0 if x < 0.5 else 1 for x in logits]
    labels = labels.cpu().numpy()
    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average="micro")
    macro_f1 = f1_score(labels, prediction, average="macro")
    return accuracy, micro_f1, macro_f1



adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(args.Branch_lst)
np.random.seed(args.seed)
torch.random.manual_seed(args.seed)

supports = preprocess_adj(adj)


device = torch.device(args.device)
train_label = torch.from_numpy(y_train).long().to(device)
features = features.to(device)
num_classes = train_label.shape[1]
train_label = torch.squeeze(train_label).float()
train_mask = torch.from_numpy(train_mask.astype(np.int)).to(device)
val_label = torch.from_numpy(y_val).long().to(device)
val_label = torch.squeeze(val_label).float()
val_mask = torch.from_numpy(val_mask.astype(np.int)).to(device)
test_label = torch.from_numpy(y_test).long().to(device)
test_label = torch.squeeze(test_label).float()
test_mask = torch.from_numpy(test_mask.astype(np.int)).to(device)

i = torch.from_numpy(supports[0]).long().to(device)
v = torch.from_numpy(supports[1]).to(device)
support = torch.sparse.FloatTensor(i.t(), v, supports[2]).float().to(device)
feat_dim = features.shape[1]


net = GCN(feat_dim, num_classes)
net.to(device)
optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

# SectorIndustry is very sensitive to this parameter
balance_weight = 3 if args.Branch_lst is None else (2.9 if 'SectorIndustry' in args.Branch_lst else 3)
pos_weight = torch.tensor([balance_weight]).to(device)
loss_fcn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')


net.train()
for epoch in range(args.epochs):
    out = net((features, support)).squeeze()
    loss = masked_loss(out, train_label, train_mask, loss_fcn)
    loss += args.weight_decay * net.l2_loss()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    mask_bool = train_mask.bool()
    out = out[mask_bool]
    eval_label = train_label[mask_bool]
    train_acc, train_micro_f1, train_macro_f1 = score(
        out, eval_label
    )
    if epoch % 10 == 0:

        print(
            "Epoch {:d} | Train accuracy {:.4f} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} ".format(
                epoch,
                train_acc,
                loss.item(),
                train_micro_f1,
                train_macro_f1,
            )
        )

net.eval()
out = net((features, support)).squeeze()
mask_bool = test_mask.bool()
out = out[mask_bool]
eval_label = test_label[mask_bool]
test_acc, test_micro_f1, test_macro_f1 = score(out , eval_label)
print(
        "Test accuracy {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f} ".format(
            test_acc,
            test_micro_f1,
            test_macro_f1,
        )
    )

print("finished")
