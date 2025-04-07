import os

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import roc_curve, auc, roc_auc_score
import seaborn as sns
import colorbm as cbm
# 读取 Excel 文件数据
df = pd.read_csv('concat.csv')

# 提取标签和模型预测结果列
labels = df['label']
# names=df.columns.tolist()[4:-1]
names=df.columns.tolist()[4:8]
print(names)
predictions = df[names]
sns.set_palette(sns.color_palette(cbm.pal('lancet').as_hex))
plt.rcParams.update({'font.size': 11})
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
# colors = ['#8ECFC9', '#FFBE7A', '#82B0D2', '#FA7F6F', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
# 绘制AUC-ROC曲线图
plt.figure()
for i, col in enumerate(predictions.columns):
    fpr, tpr, _ = roc_curve(labels, predictions[col])
    roc_auc = roc_auc_score(labels, predictions[col])
    # roc_auc = auc(fpr, tpr)
    if i == 0:
        plt.plot(fpr, tpr, label=f"{names[i]} (AUC = {roc_auc:.4f})",linewidth=1)
    elif i == 1:
        plt.plot(fpr, tpr, label=f"{names[i]} (AUC = {roc_auc:.4f})",linewidth=1)
    elif i == 2:
        plt.plot(fpr, tpr, label=f"{names[i]} (AUC = {roc_auc:.4f})",linewidth=1)
    elif i == 3:
        plt.plot(fpr, tpr, label=f"{names[i]} (AUC = {roc_auc:.4f})",linewidth=3,color="red")
    # elif i == 4:
    #     plt.plot(fpr, tpr, label=f"{names[i]} (AUC = {roc_auc:.4f})", linewidth=1)
    # if i == 0:
    #     plt.plot(fpr, tpr, label=f"A100+V100 (AUC = {roc_auc:.4f})",linewidth=1)
    # elif i == 1:
    #     plt.plot(fpr, tpr, label=f"A150+V150 (AUC = {roc_auc:.4f})",linewidth=3)
    # elif i == 2:
    #     plt.plot(fpr, tpr, label=f"A100+A150 (AUC = {roc_auc:.4f})",linewidth=1)
    # elif i == 3:
    #     plt.plot(fpr, tpr, label=f"V100+V150 (AUC = {roc_auc:.4f})", linewidth=1)
    # elif i == 4:
    #     plt.plot(fpr, tpr, label=f"A100+V100+A150+V150 (AUC = {roc_auc:.4f})",linewidth=1)
savepath=r"results_RF\ROC"
os.makedirs(savepath,exist_ok=True)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
# plt.show()
# plt.savefig("ROC_combined.tif",format="tif",dpi=600)
plt.savefig(os.path.join(savepath,"ROC_two.tif"),format="tif",dpi=600)
