import pandas as pd

with open('dev_y.txt','r',encoding='utf-8') as f:
    labels = [int(x.strip()) for x in f.readlines()]
    print(labels)


length = []
with open('train_x.txt','r',encoding='utf-8') as f:
    texts = f.readlines()

for t in texts:
    length.append(len(t.split(' ')))

dic = {'length':length}
df =pd.DataFrame(dic)
print(df.describe())
