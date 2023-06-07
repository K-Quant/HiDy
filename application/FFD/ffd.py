import pandas as pd
import numpy as np
import torchvision
import torch
from torch import nn, optim 
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import MultipleLocator
import time
import argparse


def construct_dataset(embedding_path, label_path):
    embedding_df = pd.read_csv(embedding_path)
    label_df = pd.read_csv(label_path)
    dataset_X = list()
    dataset_Y = list()

    for idx in range(len(embedding_df)):
        embedding_list = embedding_df["embeddings"][idx][1:][:-1].split(",")
        dataset_X.append(np.array(embedding_list))
    
    dataset_X = np.array(dataset_X).astype(np.float)

    for idx in range(len(label_df)):
        dataset_Y.append(label_df["label"][idx])


    dataset_Y = np.array(dataset_Y)

    dataset = dict()
    dataset["train_X"] = dataset_X
    dataset["train_Y"] = dataset_Y
    return dataset



def split_dataset(x, y):
    shape = x.shape[1]
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    dataset = torch.utils.data.TensorDataset(x, y)
    return shape, dataset



# Binary Classifcation
class singleHidden_MLP(torch.nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.fc1 = torch.nn.Linear(n_in, n_hidden)
        self.fc2 = torch.nn.Linear(n_hidden, n_out)
        #self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        #x = self.sigmoid(x)
        return x




def SingleLayerTrain(MLP, trainloader, loss_func, optimizer, num_epoch):
    loss_list = []
    num_break = 0
    for epoch in range(num_epoch):
        train_loss = 0.0
        for x, y in trainloader:
            optimizer.zero_grad()
            outs = MLP(x)
            y = torch.unsqueeze(y,1)
            loss = loss_func(outs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*x.size(0)
        train_loss = train_loss/len(trainloader.dataset)
        loss_list.append(train_loss)
        if epoch % 10 == 0:
            print("epoch {}, loss {}".format(epoch, train_loss))
        if (epoch > 150) and (abs(loss_list[epoch - 1] - train_loss) < 0.000001):
            if num_break < 2:
                num_break += 1
            else:
                break
    return




def SingleLayerTest(test_loader, MLP, loss_func):
    auc = 0.0
    train_loss = 0.0
    l = 0
    for test_x, test_y in test_loader:
        l += 1
        predicted = MLP(test_x)
        y = torch.unsqueeze(test_y, 1)
        loss = loss_func(predicted, y)
        train_loss += loss.item() * test_x.size(0)
        
        predicted = torch.sigmoid(predicted)
        predicted = torch.max(predicted, 1)
        predicted = int(predicted.values >= 0.5)
        auc += int(predicted == test_y)
    return train_loss/l, auc/l




def init_single_model(model):
    torch.nn.init.normal_(model.fc1.weight, mean = 0, std = 0.01)
    torch.nn.init.normal_(model.fc2.weight, mean = 0, std = 0.01)
    torch.nn.init.constant_(model.fc2.bias, val = 0)
    torch.nn.init.constant_(model.fc2.bias, val = 0)




# using random sub-sample cross validation (a non-exhaustive method)
def best_nerous_viaCV(trainset, loss_func, feature_shape, lr=0.00005):
    auc_max = 0
    best_nerous = 0
    nerous=10
    for h in range(1, 5):
        trainds, testds = train_test_split(trainset, test_size=0.15, random_state=23)
        trainLoader = torch.utils.data.DataLoader(trainds, batch_size=1, shuffle=True)
        testLoader = torch.utils.data.DataLoader(testds, batch_size=1, shuffle=False)
        tmp_MLP = singleHidden_MLP(feature_shape, h, 1)
        print(tmp_MLP)
        #init_single_model(tmp_MLP)
        optimizer = torch.optim.SGD(tmp_MLP.parameters(), lr, momentum=0.9)
        SingleLayerTrain(tmp_MLP, trainLoader, loss_func, optimizer, num_epoch_single)
        train_loss, train_auc = SingleLayerTest(trainLoader, tmp_MLP, loss_func)
        loss, auc = SingleLayerTest(testLoader, tmp_MLP, loss_func)
        
        train_loss_ds.append(train_loss)
        train_auc_ds.append(train_auc)
        test_loss_ds.append(loss)
        test_auc_ds.append(auc)
        
        print("="*10)
        if auc_max < auc:
            auc_max = auc
        print("Testing Accuracy: ", round(auc * 100, 2), "%")
    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, default=None, help="The path of input that you want to load.")
    args = parser.parse_args()
    
    loss_func_single = torch.nn.BCEWithLogitsLoss() # binary cross entropy loss function
    train_loss_ds = []
    train_auc_ds = []
    test_loss_ds = []
    test_auc_ds = []
    num_epoch_single = 500
    H = [] # hidden layers for each binary classification dataset
    train_loss = []
    train_auc = []
    test_loss = []
    test_auc = []
    tmpDataset = construct_dataset(args.input_path, "data/processedAudit2.csv")
    print("*"*80)
    train_x, train_y = tmpDataset['train_X'], tmpDataset['train_Y']
    shape, trainset = split_dataset(train_x, train_y)
    H.append(best_nerous_viaCV(trainset, loss_func_single, shape))





