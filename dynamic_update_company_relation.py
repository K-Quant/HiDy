from company_rel import com_rel_extract
from fusion import com_rel_fusion
from news import get_news
import os

def dynamic_pipline(start_date_str, required_days):
    dirname, filename = os.path.split(os.path.abspath(__file__)) 
    get_news.get_tushare_news_n_days("2022/12/20", 7,dirname+"/news.json")
    com_rel_extract.run_extraction([dirname+"/news.json"], dirname+"/data_extraction.json", "json")
    df = com_rel_fusion.fusion(dirname+"/data_extraction.json",dirname+"/data_fusion.csv")
    print(len(df))
    jdata = df.to_json(orient='records', force_ascii=False)
    return jdata


if __name__ == "__main__":
    dynamic_pipline("2022/12/20", 7)
