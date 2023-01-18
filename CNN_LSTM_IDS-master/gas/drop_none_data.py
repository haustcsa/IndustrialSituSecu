import pandas as pd

#
df = pd.read_csv('IanArffDataset.csv')
df = df.drop(df[df.setpoint == '?'].index)
print(df.length)
df.to_csv('Droped-Gas.csv')
# print(df.dropna(how='all'))

