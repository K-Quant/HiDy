
from data import build_corpus
from utils import extend_maps, prepocess_data_for_lstmcrf
from evaluate import hmm_train_eval, crf_train_eval, bilstm_train_and_eval
import argparse
import torch


def get_argparse():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--HMM", default=False,
                        help='Whether to do the HMM experiment')
    parser.add_argument("--BILSTM", default=False,
                        help='Whether to do the BILSTM experiment')
    parser.add_argument("--BILSTMCRF", default=False,
                        help='Whether to do the BILSTMCRF experiment')
    parser.add_argument("--IsLoadHMM", default=True, help='Whether to load a saved model without the need for retraining')
    parser.add_argument("--IsLoadBILSTM", default=False)
    parser.add_argument("--IsLoadBILSTMCRF", default=False)
    parser.add_argument("--save_HMM_path", default="./ckpts/hmm.pkl")
    parser.add_argument("--save_BILSTM_path", default="./ckpts/bilstm.pkl")
    parser.add_argument("--save_BILSTM_CRF_path", default="./ckpts/bilstm_crf.pkl")
    parser.add_argument("--device", default=torch.device("cpu"))
    return parser



if __name__ == "__main__":
    args = get_argparse().parse_args()
    print("Loading dataset......")
    train_word_lists, train_tag_lists, word2id, tag2id = \
        build_corpus("train")
    dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)

    if args.HMM == True:
        # HMM
        print("HMM...")
        hmm_pred = hmm_train_eval(
            (train_word_lists, train_tag_lists),
            (test_word_lists, test_tag_lists),
            word2id,
            tag2id,
            args = args
        )


    if args.BILSTM == True:
        # BiLSTM
        print("BiLSTM......")
        bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)
        lstm_pred = bilstm_train_and_eval(
            (train_word_lists, train_tag_lists),
            (dev_word_lists, dev_tag_lists),
            (test_word_lists, test_tag_lists),
            bilstm_word2id, bilstm_tag2id,
            crf=False,
            args = args
        )

    if args.BILSTMCRF == True:
        print("BiLSTM+CRF......")
        # 如果是加了CRF的lstm还要加入<start>和<end> (解码的时候需要用到)
        crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
        # 还需要额外的一些数据处理
        train_word_lists, train_tag_lists = prepocess_data_for_lstmcrf(
            train_word_lists, train_tag_lists
        )
        dev_word_lists, dev_tag_lists = prepocess_data_for_lstmcrf(
            dev_word_lists, dev_tag_lists
        )
        test_word_lists, test_tag_lists = prepocess_data_for_lstmcrf(
            test_word_lists, test_tag_lists, test=True
        )
        lstmcrf_pred = bilstm_train_and_eval(
            (train_word_lists, train_tag_lists),
            (dev_word_lists, dev_tag_lists),
            (test_word_lists, test_tag_lists),
            crf_word2id, crf_tag2id,
            args = args,
            crf=True
        )




