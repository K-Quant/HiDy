from multiprocessing import freeze_support

from .get_sen_set import user_input_parse
from .data_processing import prepare_company_relation, prepare_dataset, obtain_final_predictions
# from train_company_bert_softmax import train_test_mode
from .company_relation_extraction import prediction_mode
import os

dirname, filename = os.path.split(os.path.abspath(__file__))


def run_extraction(raw_file_list, output_file_path, input_mode):
    file_mode = "default"
    if input_mode != "json":
        file_mode = input_mode
    flag = user_input_parse(input_mode, raw_file_list, file_mode)
    flag = True
    if flag:
        prepare_company_relation(dirname + "/data/news_sententces_set.json", input_mode)

        '''
        # If the user choose the training and testing accuracy mode:
        prepare_dataset(divide_label=True)
        train_test_mode()
        '''

        # If the user choose the prediction mode:
        prepare_dataset()
        prediction_mode()
        obtain_final_predictions(dirname + "/data/pre_result.json", output_file_path)
        # You can just show the file "final_results.csv" to the front-end.
    else:
        print("None triples extracted!")
    return

# if __name__ == '__main__':
#     freeze_support()
#     run_extraction(["./inputs/yuncaijing_new.json"], "./data.csv", "json")
