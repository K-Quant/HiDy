import sys, json
import torch
import pandas as pd
import os
from torch import nn, optim
import numpy as np
import opennre
from opennre import encoder, model, framework
import logging
import argparse
from tqdm import tqdm



def test_file_processing(testfile_path="comp_test_file.json"):
    testset_df = pd.read_json(testfile_path)
    rel_set = []
    for idx in range(len(testset_df)):
        data = {}
        data["token"] = list(testset_df["token"][idx])
        data["h"] = testset_df["h"][idx]
        data["t"] = testset_df["t"][idx]
        data["relation"] = testset_df["relation"][idx]
        rel_set.append(data)
    file = open('comp_test.txt','w',encoding='utf-8')
    for data_item in rel_set:
        file.write(str(data_item)+'\n')
    file.close()




def train_test_mode(train_model=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_entity', action='store_true', help='Mask entity mentions')
    args = parser.parse_args()

    # Some basic settings
    sys.path.append("../..")
    if not os.path.exists('ckpt3'):
        os.mkdir('ckpt3')
    ckpt = 'ckpt3/compay_rel_pcnn_softmax.pth.tar'

    # Check data
    rel2id = json.load(open('comp_rel2id.json'))

    # # Download Glove
    # root_path = "../pretrain/"
    # #opennre.download('glove', root_path=root_path)
    # word2id = json.load(open(os.path.join(root_path, 'word2id.json')))
    # word2vec = np.load(os.path.join(root_path, 'vec.npy'))

    # # Define the sentence encoder
    # sentence_encoder = opennre.encoder.CNNEncoder(
    #     token2id=word2id,
    #     max_length=128,
    #     word_size=100,
    #     position_size=5,
    #     hidden_size=230,
    #     blank_padding=True,
    #     kernel_size=3,
    #     padding_size=1,
    #     word2vec=word2vec,
    #     dropout=0.5
    # )
    sentence_encoder = opennre.encoder.BERTEncoder(
        max_length=80, 
        pretrain_path='../pretrain/chinese_wwm_pytorch',
        mask_entity=args.mask_entity
    )

    #sentence_encoder()
    #print(sentence_encoder)

    # Define the model
    model = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
    model =model.cuda()

    # test_file_processing()
    # Define the whole training framework
    framework = opennre.framework.SentenceRE(
        train_path=None,
        val_path=None,
        test_path='comp_test.txt',
        model=model,
        ckpt=ckpt,
        batch_size=32, # Modify the batch size w.r.t. your device
        max_epoch=10,
        lr=2e-5,
        opt='adamw'
    )

    # Train the model
    if train_model:
        framework.train_model()

    # Test the model
    framework.load_state_dict(torch.load(ckpt)['state_dict'])
    result = framework.eval_model(framework.test_loader)

    acc = round(result['acc'] * 100, 1)
    print('Ours RE Prediction Accuracy: {}%'.format(acc))


train_test_mode(train_model=False)
