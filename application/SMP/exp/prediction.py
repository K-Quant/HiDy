import sys
sys.path.insert(0, sys.path[0]+"/../")
from utils.dataloader import create_test_loaders
import numpy as np
import pandas as pd
import torch
import argparse
from tqdm import tqdm
from models.model import RSR
import json


relation_model_dict = [
    'RSR'
]


def get_model(model_name):
    if model_name.upper() == 'RSR':
        return RSR

    raise ValueError('unknown model name `%s`' % model_name)


def inference(model, data_loader, stock2stock_matrix=None):
    model.eval()
    preds = []
    for i, slc in tqdm(data_loader.iter_daily(), total=data_loader.daily_length):
        # 当日切片
        feature, label, market_value, stock_index, index, mask = data_loader.get(slc)
        # feature, label, index = data_loader.get(slc)
        with torch.no_grad():
            pred = model(feature, stock2stock_matrix[stock_index][:, stock_index])
            preds.append(
                pd.DataFrame({'pred_score': pred.cpu().numpy(), 'label': label.cpu().numpy(), }, index=index))
    preds = pd.concat(preds, axis=0)
    return preds


def _prediction(param_dict, test_loader, device):
    """
    test single model first, load model from folder and do prediction
    """
    # test_loader = create_test_loaders(args, for_individual=for_individual)
    # stock2concept_matrix = param_dict['stock2concept_matrix']
    stock2stock_matrix = param_dict['stock2stock_matrix']
    print('load model ', param_dict['model_name'])


    stock2stock_matrix = torch.Tensor(np.load(stock2stock_matrix)).to(device)
    num_relation = stock2stock_matrix.shape[2]  # the number of relations
    model = get_model(param_dict['model_name'])(num_relation=num_relation, d_feat=param_dict['d_feat'],
                                                num_layers=param_dict['num_layers'])

    model.to(device)

    model.load_state_dict(torch.load(param_dict['model_dir'] + '/model.bin', map_location=device))
    print('predict in ', param_dict['model_name'])
    pred = inference(model, test_loader, stock2stock_matrix)
    return pred


def prediction(args, model_path, device):
    param_dict = json.load(open(model_path+'/info.json'))['config']
    param_dict['model_dir'] = model_path
    test_loader = create_test_loaders(args, param_dict, device=device)
    pred = _prediction(param_dict, test_loader, device)
    return pred


def main(args, device):
    model_path = args.model_path
    pd.to_pickle(prediction(args, model_path, device), args.pkl_path)


def parse_args():
    """
    deliver arguments from json file to program
    :param param_dict: a dict that contains model info
    :return: no return
    """
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--test_start_date', default='2019-01-01')
    parser.add_argument('--test_end_date', default='2022-12-31')
    parser.add_argument('--target', default='t+0')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--pkl_path', default=None,
                        help='location to save the pred dictionary file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    model_dict_list = [
        './pretrain/RSR_hidy',
        './pretrain/RSR_doc2edge_is',
        './pretrain/RSR_dueefin_is', './pretrain/RSR_fr2kg_is',
        './pretrain/RSR_is', './pretrain/RSR_sht_is'
    ]
    pkl_path_list = [
        './output/hidy.pkl',
        './output/doc2edga.pkl',
        './output/dueefin.pkl', './output/fr2kg.pkl',
        './output/is.pkl', './output/sht.pkl'
    ]

    for i in range(len(model_dict_list)):
        model_path = model_dict_list[i]
        pkl_path = pkl_path_list[i]
        pd.to_pickle(prediction(args, model_path, device), pkl_path)
        print('finish inference '+ model_path)
