import pandas as pd


df = pd.read_csv('train.csv')
text = df.iloc[:,-1].values.tolist()
length = [len(t.split()) for t in text]

dic = {'len':length}
df = pd.DataFrame(dic)
print(df.describe())
