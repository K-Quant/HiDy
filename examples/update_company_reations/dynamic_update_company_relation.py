import sys

sys.path.append('../../../fusion')
sys.path.append('../../../company_rel')

from company_rel import com_rel_extract
from fusion import com_rel_fusion
# from news import get_news


def dynamic_pipline( original_news_path, extracted_data_path, fusion_data_path):
    com_rel_extract.run_extraction(original_news_path, extracted_data_path, "json")
    com_rel_fusion.fusion(extracted_data_path, fusion_data_path)


if __name__ == "__main__":
    dynamic_pipline("2022/12/20", 7, "./news.json", "./data_extraction.json", "./data_fusion.csv")
