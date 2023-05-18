import pandas as pd

relation_dict =['same_industry','rise','compete','unknown','cooperate','increase_holding','fall','supply'
    ,'be_reduced_holding','be_invested','reduce_holding','superior','be_increased_holding','subordinate','invest','dispute','be_supplied']

# 返回的是一个DataFrame数据
df = pd.read_csv("./fusionResult.csv")

for rela in relation_dict:
    # print(df)
    nd = df.loc[df['relation'] == rela]
    nd.to_csv('./result/2022-12-17/'+rela+'_22_12_17.csv')
