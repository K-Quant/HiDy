from DART import *

## set path for data source
path_dict = dict()
path_dict["yuncaijing"] = './Data/eventnode_roberta_ycj.csv'
path_dict["10jqka"] = './Data/eventnode_roberta_10jqka.csv'
path_dict["eastmoney"] = './Data/eventnode_roberta_eastmoney.csv'
path_dict["company"] = './Data/company_alias_stock_dict_qcc.json'
path_dict["testset"] = './Data/test.csv'
path_dict["fewshot"] = './Data/fewshot.csv'
path_dict["triples"] = './Data/test.csv'
path_dict["main_domain"] = './Data/main_domain_dict.csv'
domain_list = ["财经/交易", "产品行为", "交往", "竞赛行为", "人生", "司法行为", "灾害/意外", "组织关系",  "组织行为"]
srcs_list = ["yuncaijing", "10jqka", "eastmoney"]

## init recall and sp for DART (extension)
fewShot_df = pd.read_csv(path_dict["fewshot"])
init_recall, init_sp = init_recall_sp(fewShot_df)

## run DART algorithm
dart = DART(path_dict, srcs_list, domain_list, init_recall, init_sp, test_mode=True)
dart.run()

