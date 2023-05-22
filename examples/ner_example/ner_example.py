import sys

sys.path.append('../../ner')

from ner.NER import NER

if __name__ == '__main__':
    # annual report
    NER(data_dir='AnnualReportData/', entities=['product'], epoch=1, whether_train_evaluation=False,
        output_dir='output/meso_ar_KB.txt')
    # industrial strategy
    NER(data_dir='StrategyData/', entities=['industry'], epoch=1, whether_train_evaluation=False,
        output_dir='output/macro_s_KB.txt')
