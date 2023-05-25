import argparse
from pprint import pprint
from paddlenlp import Taskflow
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import json


def find_chinese(file):
    chinese = re.findall('[\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b\u4e00-\u9fa5]', file)
    sentence = ""
    for item in chinese:
        sentence += item
    return sentence



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, default=None, help="The path of input text data that you want to load.")
    parser.add_argument("--output_path", type=str, default=None, help="The path of extracted knowledge.")
    args = parser.parse_args()
    
    df = pd.read_csv(args.input_path,header=None)
    sen_dict = {}
    for idx in tqdm(range(len(df))):
        sentence = df[1][idx]
        item = df[0][idx]
        # sentence = find_chinese(sentence)
        sen_dict[item] = sentence



    schema = ['产品', '原材料'] # Define the schema for entity extraction
    ie = Taskflow('information_extraction', schema=schema)

    final_results = {}
    for item in tqdm(sen_dict.keys()):
        sentence = sen_dict[item]
        if len(sentence) > 0:
            extract_dict = ie(sentence)
            if "产品" in extract_dict[0] and "原材料" in extract_dict[0]:
                name = extract_dict[0]["产品"][0]["text"]
                final_results[name] = []
                for idx in range(len(extract_dict[0]["原材料"])):
                    final_results[name].append(extract_dict[0]["原材料"][idx]["text"])

    time = (args.input_path).split(".")[0]
    tuples = dict()
    tuples["head"] = list()
    tuples["tail"] = list()
    for key in final_results.keys():
        for item in final_results[key]:
            tuples["head"].append(key)
            tuples["tail"].append(item)
    tuples["relation"] = "downstream"
    tuples["time"] = time
    my_df = pd.DataFrame(tuples)
    my_df.to_csv(args.output_path, index=False)