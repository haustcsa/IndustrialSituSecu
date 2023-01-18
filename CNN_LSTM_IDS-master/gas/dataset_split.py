from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.utils import shuffle

# dataset = pd.read_csv('4-Droped-Gas.csv')
dataset = pd.read_csv('4-Droped-Gas.csv')
data = shuffle(dataset, random_state=42)
# data = data.drop([0])
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
train_set.to_csv('15-gas_tr.csv', index=None)
test_set.to_csv('15-gas_te.csv', index=None)