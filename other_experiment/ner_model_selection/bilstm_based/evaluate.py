import time
from collections import Counter

from models.hmm import HMM
from models.crf import CRFModel
from models.bilstm_crf import BILSTM_Model
from utils import save_model, flatten_lists
from evaluating import Metrics
import pickle
import torch

def hmm_train_eval(train_data, test_data, word2id, tag2id, args, remove_O=False):
    """训练并评估hmm模型"""
    train_word_lists, train_tag_lists = train_data
    test_word_lists, test_tag_lists = test_data

    hmm_model = HMM(len(tag2id), len(word2id))
    if args.IsLoadHMM:
        print("Loading HMM......")
        with open(args.save_HMM_path, 'rb') as model:
            hmm_model = pickle.load(model)
    else:
        print("Training HMM......")
        hmm_model.train(train_word_lists,
                        train_tag_lists,
                        word2id,
                        tag2id)
        print("Saving HMM......")
        save_model(hmm_model, args.save_HMM_path)

    print("***** Testing ******")
    pred_tag_lists = hmm_model.test(test_word_lists,
                                    word2id,
                                    tag2id)
    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()
    return pred_tag_lists


def crf_train_eval(train_data, test_data, args, remove_O=False):
    train_word_lists, train_tag_lists = train_data
    test_word_lists, test_tag_lists = test_data
    crf_model = CRFModel()
    if args.IsLoadCRF:
        print("Loading CRF......")
        with open(args.save_CRF_path, 'rb') as model:
            crf_model = pickle.load(model)
    else:
        print("Training CRF......")
        crf_model.train(train_word_lists, train_tag_lists)
        save_model(crf_model, args.save_CRF_path)

    print("***** Testing ******")
    pred_tag_lists = crf_model.test(test_word_lists)
    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()
    return pred_tag_lists


def bilstm_train_and_eval(train_data, dev_data, test_data,
                          word2id, tag2id, args, crf=True, remove_O=False):
    train_word_lists, train_tag_lists = train_data
    dev_word_lists, dev_tag_lists = dev_data
    test_word_lists, test_tag_lists = test_data

    start = time.time()
    vocab_size = len(word2id)
    out_size = len(tag2id)
    bilstm_model = BILSTM_Model(vocab_size, out_size, args=args, crf=crf)
    model_name = "bilstm_crf" if crf else "bilstm"
    if  crf == False:
        if args.IsLoadBILSTM:
            print("Loading BiLSTM......")
            with open(args.save_BILSTM_path, 'rb') as model:
                bilstm_model = pickle.load(model)
        else:
            print("Training BiLSTM......")
            bilstm_model.train(train_word_lists, train_tag_lists,
                               dev_word_lists, dev_tag_lists, word2id, tag2id, args, crf)
            # load the best model
            with open(args.save_BILSTM_path, 'rb') as model:
                bilstm_model = pickle.load(model)
    else:
        if args.IsLoadBILSTMCRF:
            print("Loading BiLSTM+CRF......")
            with open(args.save_BILSTM_CRF_path, 'rb') as model:
                bilstm_model = pickle.load(model)
        else:
            print("Training BiLSTM+CRF......")
            bilstm_model.train(train_word_lists, train_tag_lists,
                               dev_word_lists, dev_tag_lists, word2id, tag2id, args, crf)
            # load the best model
            with open(args.save_BILSTM_CRF_path, 'rb') as model:
                bilstm_model = pickle.load(model)

    print("***** Testing ******")
    # torch.save(bilstm_model.state_dict(), "./ckpts/"+model_name + '.pt')
    # bilstm_model.load_state_dict(torch.load(output_path + "./ckpts/"+model_name + '.pt'))
    print("finished training, total {} seconds.".format(int(time.time()-start)))
    pred_tag_lists, test_tag_lists = bilstm_model.test(
        test_word_lists, test_tag_lists, word2id, tag2id, args)
    # print("pred_tag_lists", pred_tag_lists)
    # print("test_tag_lists", test_tag_lists)
    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

    return pred_tag_lists



