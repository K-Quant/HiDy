import  numpy as np
import  pickle as pkl
import  networkx as nx
import  scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import pickle
import pandas as pd
import torch
import  sys


def parse_index_file(filename):
    """
    Parse index file.
    """
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """
    Create mask.
    """
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(Branch_lst):
    branch_dic = {'IndustryChain': ['adj_CPC', 'supply'],
                  'SectorIndustry': ['adj_CIC', 'same_industry', 'superior'],
                  'Ownership': ['invest', 'increase_holding', 'be_increased_holding', 'be_reduced_holding',
                                'be_invested',
                                'reduce_holding'],
                  'Partnership': ['dispute', 'fall', 'cooperate', 'rise']}

    filename = 'sparse_matrices.pkl'
    with open(filename, 'rb') as file:
        matrix_dict = pickle.load(file)

    labels = matrix_dict['label']

    train_mask = matrix_dict['train_mask']
    val_mask = matrix_dict['val_mask']
    test_mask = matrix_dict['test_mask']

    adj_homo = matrix_dict['adj_CFC'] + matrix_dict['adj_CAC']  # CSMAR
    if Branch_lst != None:
        for i in Branch_lst:
            for j in branch_dic[i]:
                # print(j, "start")
                adj_homo = adj_homo + matrix_dict[j]
    adj_homo.data = np.ones_like(adj_homo.data)
    matrix_dict['adj_homo'] = adj_homo

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

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

    return matrix_dict['adj_homo'], dataset_X, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """
    Convert sparse matrix to tuple representation.
    """
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """
    Row-normalize feature matrix and convert to tuple representation
    """
    rowsum = np.array(features.sum(1)) # get sum of each row, [2708, 1]
    r_inv = np.power(rowsum, -1).flatten() # 1/rowsum, [2708]
    r_inv[np.isinf(r_inv)] = 0. # zero inf data
    r_mat_inv = sp.diags(r_inv) # sparse diagonal matrix, [2708, 2708]
    features = r_mat_inv.dot(features) # D^-1:[2708, 2708]@X:[2708, 2708]
    return sparse_to_tuple(features) # [coordinates, data, shape], []


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)) # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() # D^-0.5AD^0.5


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)





def chebyshev_polynomials(adj, k):
    """
    Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation).
    """
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)
