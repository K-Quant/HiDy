import numpy as np
import scipy.sparse as sp
import torch
import pickle
import pandas as pd

def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(Branch_lst):
    branch_dic = {'IndustryChain': ['adj_CPC', 'supply'],
                  'SectorIndustry': ['adj_CIC', 'same_industry', 'superior'],
                  'Ownership': ['invest', 'increase_holding', 'be_increased_holding', 'be_reduced_holding',
                                'be_invested',
                                'reduce_holding'],
                  'Partnership': ['dispute', 'fall', 'cooperate', 'rise']}

    # 读取之前保存的.pkl文件
    filename = './sparse_matrices.pkl'
    with open(filename, 'rb') as file:
        matrix_dict = pickle.load(file)


    adj_homo = matrix_dict['adj_CFC'] + matrix_dict['adj_CAC']  # CSMAR
    if Branch_lst != None:
        for i in Branch_lst:
            for j in branch_dic[i]:
                adj_homo = adj_homo + matrix_dict[j]
    adj_homo.data = np.ones_like(adj_homo.data)
    adj_homo = adj_homo.tocoo()
    adj_homo = torch.FloatTensor(np.array(adj_homo.todense()))

    # loading feature
    if Branch_lst == None:
        embedding_path = '../features/CSMAR.csv'
    elif len(Branch_lst) == 1:
        if Branch_lst == ['IndustryChain']:
            embedding_path = '../features/CSMAR+IndustryChain.csv'
        elif Branch_lst == ['SectorIndustry']:
            embedding_path = '../features/CSMAR+SectorIndustry.csv'
        elif Branch_lst == ['Ownership']:
            embedding_path = '../features/CSMAR+Ownership.csv'
        elif Branch_lst == ['Partnership']:
            embedding_path = '../features/CSMAR+Partnership.csv'
        else:
            print("ERROR Branch_lst")
            exit()
    elif len(Branch_lst) == 2:
        if 'IndustryChain' in Branch_lst and 'SectorIndustry' in Branch_lst:
            embedding_path = '../features/CSMAR+IndustryChain+SectorIndustry.csv'
        elif 'IndustryChain' in Branch_lst and 'Ownership' in Branch_lst:
            embedding_path = '../features/CSMAR+IndustryChain+Ownership.csv'
        elif 'IndustryChain' in Branch_lst and 'Partnership' in Branch_lst:
            embedding_path = '../features/CSMAR+IndustryChain+Partnership.csv'
        elif 'Ownership' in Branch_lst and 'SectorIndustry' in Branch_lst:
            embedding_path = '../features/CSMAR+Ownership+SectorIndustry.csv'
        elif 'Partnership' in Branch_lst and 'SectorIndustry' in Branch_lst:
            embedding_path = '../features/CSMAR+Partnership+SectorIndustry.csv'
        elif 'Partnership' in Branch_lst and 'Ownership' in Branch_lst:
            embedding_path = '../features/CSMAR+Ownership+Partnership.csv'
        else:
            print("ERROR Branch_lst")
            exit()
    elif len(Branch_lst) == 3:
        if 'IndustryChain' in Branch_lst and 'SectorIndustry' in Branch_lst and 'Ownership' in Branch_lst:
            embedding_path = '../features/CSMAR+Ownership+SectorIndustry+IndustryChain.csv'
        elif 'IndustryChain' in Branch_lst and 'SectorIndustry' in Branch_lst and 'Partnership' in Branch_lst:
            embedding_path = '../features/CSMAR+Partnership+SectorIndustry+IndustryChain.csv'
        elif 'Ownership' in Branch_lst and 'SectorIndustry' in Branch_lst and 'Partnership' in Branch_lst:
            embedding_path = '../features/CSMAR+Partnership+SectorIndustry+Ownership.csv'
        elif 'Ownership' in Branch_lst and 'IndustryChain' in Branch_lst and 'Partnership' in Branch_lst:
            embedding_path = '../features/CSMAR+Partnership+IndustryChain+Ownership.csv'
        else:
            print("ERROR Branch_lst")
            exit()
    elif len(Branch_lst) == 4:
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

    return adj_homo, dataset_X, torch.tensor(matrix_dict['label']), matrix_dict['idx_train'].to(torch.long), matrix_dict['idx_val'].to(torch.long), matrix_dict['idx_test'].to(torch.long)



def normalize_adj(mx):
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

