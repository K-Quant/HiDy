## import DART package
from .dart import *
import os

def fusion(input_file_path,output_file_path):
    ## set path for data source
    path_dict = dict()
    dirname, filename = os.path.split(os.path.abspath(__file__)) 
    
    path_dict["yuncaijing"] = dirname+'/Data/eventnode_roberta_ycj.csv'
    path_dict["10jqka"] = dirname+'/Data/eventnode_roberta_10jqka.csv'
    path_dict["eastmoney"] = dirname+'/Data/eventnode_roberta_eastmoney.csv'
    path_dict["company"] = dirname+'/Data/company_alias_stock_dict_qcc.json'
    path_dict["testset"] = dirname+'/Data/annotated_relation.csv'
    path_dict["fewshot"] = dirname+'/Data/fewshot.csv'
    path_dict["triples"] = input_file_path
    path_dict["main_domain"] = dirname+'/Data/main_domain_dict.csv'
    domain_list = ["财经/交易", "产品行为", "交往", "竞赛行为", "人生", "司法行为", "灾害/意外", "组织关系",  "组织行为"]
    srcs_list = ["yuncaijing", "10jqka", "eastmoney"]


    ## init recall and sp for DART (extension)
    fewShot_df = pd.read_csv(path_dict["fewshot"])
    init_recall, init_sp = init_recall_sp(fewShot_df)

    test_df = pd.read_csv(dirname+'/Data/annotated_relation.csv')

    ## run DART algorithm
    dart = DART(path_dict, srcs_list, domain_list, init_recall, init_sp, test_mode=False)
    dart.run()

    objects_dict = dart.objects_dict

    out_knowledge = dict()
    out_knowledge["head_code"] = []
    out_knowledge["tail_code"] = []
    out_knowledge["relation"] = []
    out_knowledge["time"] = []

    relation_dict = {
        '同行':'same_industry',
        '同涨':'rise',
        '竞争':'compete',
        '未知Unknown':'unknown', 
        '合作':'cooperate', 
        '增持':'increase_holding', 
        '同跌':'fall', 
        '供应':'supply', 
        '被减持':'be_reduced_holding',
        '被投资':'be_invested', 
        '减持':'reduce_holding', 
        '上级':'superior', 
        '被增持':'be_increased_holding', 
        '下级':'subordinate', 
        '投资':'invest', 
        '纠纷':'dispute', 
        '被供应':'be_supplied'
    }

    for obj_key in tqdm(objects_dict.keys()):
        for idx in range(len(objects_dict[obj_key]["triples_label"])):
            if objects_dict[obj_key]["triples_label"][idx] == 1:
                out_knowledge["head_code"].append(objects_dict[obj_key]["triples"][idx][0])
                out_knowledge["tail_code"].append(objects_dict[obj_key]["triples"][idx][1])
                out_knowledge["relation"].append(relation_dict[objects_dict[obj_key]["triples"][idx][2]])
                out_knowledge["time"].append(objects_dict[obj_key]["datetime"][idx])

    my_df = pd.DataFrame(out_knowledge)
    my_df.to_csv(output_file_path, index=False)
    return my_df

# my_df = fusion("data.csv","fusion.csv")
# my_df.to_csv("fusion.csv", index=False)