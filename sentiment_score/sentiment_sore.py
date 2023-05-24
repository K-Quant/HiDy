import pandas as pd
import re
import json
import sys
import warnings
from tqdm.notebook import tqdm
import time

if not sys.warnoptions:
    warnings.simplefilter('ignore')
from hanlp_restful import HanLPClient
from paddlenlp import Taskflow

auth_key = "ODczQGJicy5oYW5scC5jb206bHh1Rk1ZREhaZEpieDNLNw=="
HanLP = HanLPClient('https://www.hanlp.com/api', auth=auth_key, language='zh')  # auth不填则匿名，zh中文，mul多语种
pattern = r';|\'|`|\?|\~|!|@|#|\^|&|=|\_|。|；|·|！|？|…'
sp_pattern = r'，|-|\{|\}|<|>|【|】|\+|,'
ssp_pattern = r'：|:|、'
senta = Taskflow("sentiment_analysis", model="skep_ernie_1.0_large_ch")


def divide_sen(sentence):
    sentence = sentence.replace(" ", "")
    result_list = re.split(pattern, sentence)
    for sen in result_list:
        if len(sen) > 300:
            temp_list = re.split(sp_pattern, sen)
            for temp in temp_list:
                result_list.append(temp)
            result_list.remove(sen)
    for sen in result_list:
        if len(sen) > 300:
            temp_list = re.split(ssp_pattern, sen)
            for temp in temp_list:
                result_list.append(temp)
            result_list.remove(sen)
    return result_list


def research_report_ner(begin_point, report_data):
    res = []
    ner_index = 0
    for k in tqdm(range(begin_point, len(report_data))):
        report = report_data[k]
        sentences = divide_sen(report[7])
        valid_sen = []
        temp_res = []
        for i in range(len(sentences)):
            if sentences[i] != "" and len(sentences[i]) < 150:
                valid_sen.append(sentences[i])
            if (len(valid_sen) >= 50) or (len(valid_sen) > 0 and i == len(sentences) - 1):
                ner_index += 1
                time.sleep(0.4)
                if ner_index == 1:
                    begin = time.time()
                if ner_index == 60:
                    end = time.time()
                    sleep_time = 65 - (end - begin)
                    if sleep_time < 0:
                        sleep_time = 1
                    time.sleep(sleep_time)
                    ner_index = 0
                time.sleep(0.5)
                ner_res = HanLP(valid_sen, tasks='ner')
                for j in range(len(valid_sen)):
                    if len(ner_res["ner/msra"][j]) != 0:
                        valid_entity = []
                        for entity in ner_res["ner/msra"][j]:
                            if entity[1] == "ORGANIZATION":
                                valid_entity.append(entity[0])
                        if len(valid_entity) != 0:
                            temp_res.append({
                                "researcher": report[0],
                                "orgSName": report[1],
                                "orgCode": report[2],
                                "industryName": report[3],
                                "industryCode": report[4],
                                "publishDate": report[5],
                                "title": report[6],
                                "sentence": valid_sen[j],
                                "orgEntity": valid_entity
                            }
                            )
                valid_sen = []
        for res_item in temp_res:
            res.append(res_item)
    return res


def research_report_ema(df_report):
    temp_res = []
    report_data = df_report.values
    sentence = report_data[:, 7]
    sentence = sentence.tolist()
    ema = senta(sentence)
    for i in range(len(report_data)):
        temp_res.append({
            "researcher": report_data[i][0],
            "orgSName": report_data[i][1],
            "orgCode": report_data[i][2],
            "industryName": report_data[i][3],
            "industryCode": report_data[i][4],
            "publishDate": report_data[i][5],
            "title": report_data[i][6],
            "sentence": report_data[i][7],
            "orgEntity": report_data[i][8],
            'ema_label': ema[i]['label'],
            'ema_score': ema[i]['score']
        })
    return temp_res


def extract_sentiment_score(input_path, output_path):
    research_report = json.load(open(input_path, 'r', encoding='UTF-8'))
    df = pd.DataFrame.from_dict(research_report, orient='columns')
    report_data = df.values
    res = []
    for s in tqdm(range((len(report_data) // 20) * 19, (len(report_data) // 20) * 20)):
        valid_sen = []
        sentences = divide_sen(report_data[s][9])

        for i in range(len(sentences)):
            if sentences[i] != "" and len(sentences[i]) < 150:
                valid_sen.append(sentences[i])
            if len(valid_sen) > 0 and i == len(sentences) - 1:
                time.sleep(1)
                ner_res = HanLP(valid_sen, tasks='ner')
                for j in range(len(valid_sen)):
                    if len(ner_res["ner/msra"][j]) != 0:
                        valid_entity = []
                        for entity in ner_res["ner/msra"][j]:
                            if entity[1] == "ORGANIZATION":
                                valid_entity.append(entity[0])
                        if len(valid_entity) != 0:
                            res.append({
                                "researcher": report_data[s][0],
                                "orgSName": report_data[s][1],
                                "orgCode": report_data[s][2],
                                "industryName": report_data[s][3],
                                "industryCode": report_data[s][4],
                                "publishDate": report_data[s][7],
                                "title": report_data[s][8],
                                "sentence": valid_sen[j],
                                "orgEntity": valid_entity
                            }
                            )
        ner_df = pd.DataFrame.from_dict(res, orient='columns')
        temp_res = research_report_ema(ner_df)
        with open(output_path, 'w', encoding='utf8') as f:
            json.dump(temp_res, f, indent=1, ensure_ascii=False)
