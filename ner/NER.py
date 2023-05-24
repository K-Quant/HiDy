# industrial strategy and industry_chain-company_produce_product KB

# import sys
# sys.path.append('/usr/local/lib/python3.9/site-packages')
import numpy as np
from sklearn.model_selection import ShuffleSplit
from data_utils import Documents, Dataset, SentenceExtractor, make_predictions
from data_utils import Evaluator
from models import build_lstm_crf_model
from gensim.models import Word2Vec
import os
import time
import argparse



def NER(data_dir, entities, epoch = 10, whether_train_evaluation = False, output_dir = 'output/'):
    workdir = os.listdir(data_dir)
    if '.DS_Store' in workdir:
      os.remove('./'+data_dir+'.DS_Store')

    ent2idx = dict(zip(entities, range(1, len(entities) + 1)))
    idx2ent = dict([(v, k) for k, v in ent2idx.items()])

    docs = Documents(data_dir=data_dir)
    rs = ShuffleSplit(n_splits=1, test_size=26, random_state=2023)
    train_doc_ids, test_doc_ids = next(rs.split(docs))
    train_docs, test_docs = docs[train_doc_ids], docs[test_doc_ids]

    num_cates = max(ent2idx.values()) + 1
    sent_len = 64
    vocab_size = 3000
    emb_size = 100
    sent_pad = 10
    sent_extrator = SentenceExtractor(window_size=sent_len, pad_size=sent_pad)
    train_sents = sent_extrator(train_docs)
    test_sents = sent_extrator(test_docs)

    train_data = Dataset(train_sents, cate2idx=ent2idx)
    train_data.build_vocab_dict(vocab_size=vocab_size)

    test_data = Dataset(test_sents, word2idx=train_data.word2idx, cate2idx=ent2idx)
    vocab_size = len(train_data.word2idx)

    w2v_train_sents = []
    for doc in docs:
        w2v_train_sents.append(list(doc.text))
    w2v_model = Word2Vec(w2v_train_sents, vector_size=emb_size)

    w2v_embeddings = np.zeros((vocab_size, emb_size))
    for char, char_idx in train_data.word2idx.items():
        if char in w2v_model.wv:
            w2v_embeddings[char_idx] = w2v_model.wv[char]

    seq_len = sent_len + 2 * sent_pad
    model = build_lstm_crf_model(num_cates, seq_len=seq_len, vocab_size=vocab_size,
                                 model_opts={'emb_matrix': w2v_embeddings, 'emb_size': 100, 'emb_trainable': False})
    model.summary()
    train_X, train_y = train_data[:]


    """### training"""
    model.fit(train_X, train_y, batch_size=64, epochs=epoch)

    if whether_train_evaluation:
        """### evaluation"""
        preds_train = model.predict(train_X, batch_size=64, verbose=True)
        pred_docs_train = make_predictions(preds_train, train_data, sent_pad, docs, idx2ent)
        print(pred_docs_train)
        f_score, precision, recall = Evaluator.f1_score(train_docs, pred_docs_train)
        print('train f_score: ', f_score)
        print('train precision: ', precision)
        print('train recall: ', recall)

    test_X, test_y = test_data[:]
    preds_test = model.predict(test_X, batch_size=64, verbose=True)
    pred_docs_test = make_predictions(preds_test, test_data, sent_pad, docs, idx2ent)
    print(pred_docs_test)
    f_score, precision, recall = Evaluator.f1_score(test_docs, pred_docs_test)
    print('test: f_score: ', f_score)
    print('test: precision: ', precision)
    print('test: recall: ', recall)


    # # output KB and visualize
    # # http://localhost:5000/
    if output_dir != None:
        if data_dir == './data/AnnualReportData/':
            ini_port = 5000
            file_write_obj = open(output_dir, 'w')
            for j in list(pred_docs_test.keys()):
                while True:
                    try:
                        # time.sleep(2)
                        # print("{} start".format(j))
                        ini_port += 1
                        industry_list = pred_docs_test[j]._repr_html_(portid=ini_port)
                        industry_list = list(set(industry_list))
                        for i in industry_list:
                            file_write_obj.write(j.split('-')[0])
                            file_write_obj.write('\t')
                            file_write_obj.write('mention')
                            file_write_obj.write('\t')
                            file_write_obj.write(i)
                            file_write_obj.write('\t')

                            file_write_obj.write(j.split('-')[1][:4])
                            file_write_obj.write('\n')
                        break
                    except (OSError):
                        inp = input("retry:0\nnext:1")  # Get the input
                        if inp == '1':
                            break
                        else:
                            continue

        elif data_dir == "./data/StrategyData/":
            f = open("./data/strategydata_time.txt",encoding='utf-8')            # obtain timestamps
            dic_time = {}
            while True:
                line = f.readline()
                if line:
                    dic_time[line.split('\t')[0]] = line.split('\t')[1][:-1]
                else:
                    break
            f.close()

            file_write_obj = open(output_dir, 'w')
            ini_port = 5000
            for j in list(pred_docs_test.keys()):
                while True:
                    try:
                        # time.sleep(2)
                        # print("{} start".format(j))
                        ini_port += 1
                        industry_list = pred_docs_test[j]._repr_html_(portid=ini_port)
                        industry_list = list(set(industry_list))
                        for i in industry_list:
                            file_write_obj.write(j)
                            file_write_obj.write('\t')
                            file_write_obj.write('mention')
                            file_write_obj.write('\t')
                            file_write_obj.write(i)
                            file_write_obj.write('\t')
                            if j in dic_time.keys():
                                file_write_obj.write(dic_time[j])
                            else:
                                file_write_obj.write('none')
                            file_write_obj.write('\n')
                        break

                    except (OSError):
                        inp = input("retry:0\nnext:1")  # Get the input
                        if inp == '1':
                            break
                        else:
                            continue
            if whether_train_evaluation != True:
                preds_train = model.predict(train_X, batch_size=64, verbose=True)
                pred_docs_train = make_predictions(preds_train, train_data, sent_pad, docs, idx2ent)
            for j in list(pred_docs_train.keys()):
                while True:
                    try:
                        # time.sleep(2)
                        # print("{} start".format(j))
                        ini_port += 1
                        industry_list = pred_docs_train[j]._repr_html_(portid=ini_port)
                        industry_list = list(set(industry_list))
                        for i in industry_list:
                            file_write_obj.write(j)
                            file_write_obj.write('\t')
                            file_write_obj.write('mention')
                            file_write_obj.write('\t')
                            file_write_obj.write(i)
                            file_write_obj.write('\t')
                            if j in dic_time.keys():
                                file_write_obj.write(dic_time[j])
                            else:
                                file_write_obj.write('none')
                            file_write_obj.write('\n')
                        break

                    except (OSError):
                        inp = input("retry:0\nnext:1")  # Get the input
                        if inp == '1':
                            break
                        else:
                            continue
            file_write_obj.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="./data/StrategyData/",
                        help="Load the data. for industial strategy: ./data/StrategyData/ ,for annual report: ./data/AnnualReportData/")
    parser.add_argument("--entities", type=str, default='industry',
                        help="Please provide the list of name of the entity. for industial strategy: 'industry', for annual report: 'product'")
    parser.add_argument("--output_path", type=str, default='./output/macro_s_KB.txt', help="The path of extracted knowledge. for industial strategy: ./output/macro_s_KB.txt, for annual report: ./output/meso_ar_KB.txt")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--whether_train_evaluation", type=bool, default=False, help="Whether do the evaluation for the training process.")

    args = parser.parse_args()

    NER(data_dir = args.input_path, entities = [args.entities], epoch = args.epochs, whether_train_evaluation = args.whether_train_evaluation, output_dir = args.output_path)
