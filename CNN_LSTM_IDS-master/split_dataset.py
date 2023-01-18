
import pandas as pd
import numpy as np

# 读取文件
from sklearn.utils import shuffle

df1 = pd.read_csv(r"KDDTrain+.csv")
df1["来自文件"] = "文件-R"
df2 = pd.read_csv(r"KDDTest+.csv")
df2["来自文件"] = "文件-E"
# 合并
df = pd.concat([df1, df2])
df.drop_duplicates()  # 数据去重
# 保存合并后的文件
df = shuffle(df, random_state=42)
df.to_csv('KDD.csv', encoding='utf-8')


kdd = pd.read_csv(r'Pearson-KDD\KDD-class-number.csv')
kdd_train = kdd.iloc[0:125973, :]
kdd_test = kdd.iloc[125974:, :]

kdd_train.to_csv(r'Pearson-KDD\kdd_train_header.csv', index=None)
kdd_test.to_csv(r'Pearson-KDD\kdd_test_header.csv', index=None)
# ga = pd.read_csv(r'gas/10_gas_final.arff.csv')
# ga = shuffle(ga, random_state=42)
# ga_tr = ga.iloc[0:9557, :]
# ga_te = ga.iloc[9557:, :]
# ga_tr.to_csv('gas/ga_train.csv', index=None)
# ga_te.to_csv('gas/ga_test.csv', index=None)
