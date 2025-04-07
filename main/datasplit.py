import pandas as pd
import numpy as np
np.random.seed(42)

df=pd.read_csv("data.csv")

patient_stats = df.groupby('patient_id').agg(
    node_count=('label', 'count'),
    pos_count=('label', 'sum'),
    neg_count=('label', lambda x: (x == 0).sum())
).reset_index()

shuffled = patient_stats.sample(frac=1, random_state=42).reset_index(drop=True)
target_nodes = int(len(df) * 0.7)
train_patients = []
nodes_so_far = 0

for _, row in shuffled.iterrows():
    if nodes_so_far + row['node_count'] <= target_nodes:
        train_patients.append(row['patient_id'])
        nodes_so_far += row['node_count']
    else:
        break

# Step 4: 分配训练集和测试集
train_df = df[df['patient_id'].isin(train_patients)]
test_df = df[~df['patient_id'].isin(train_patients)]

train_df.to_csv("train.csv")
test_df.to_csv("test.csv")