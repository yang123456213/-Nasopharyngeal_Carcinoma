import os
# https://pubmed.ncbi.nlm.nih.gov/38617761/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
import colorbm as cbm
from matplotlib import rcParams
sns.set_palette(sns.color_palette(cbm.pal('lancet').as_hex))
plt.rcParams.update({'font.size': 11})
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
# 从Excel文件中读取数据
from sklearn.metrics import brier_score_loss
def brief():
    data = pd.read_csv('concat.csv')

    # 选择标准值列
    standard_values = data['label'].values
    # names=data.columns.tolist()[4:8]
    names=data.columns.tolist()[:]
    print(names)

    # 选择实际测量值列
    measured_values_1 = data.loc[:,names[0]].values
    measured_values_2 = data.loc[:,names[1]].values
    measured_values_3 = data.loc[:,names[2]].values
    measured_values_4 = data.loc[:,names[3]].values
    measured_values_5 = data.loc[:,names[4]].values
    measured_values_6 = data.loc[:,names[5]].values
    measured_values_7 = data.loc[:,names[6]].values
    measured_values_8 = data.loc[:,names[7]].values
    measured_values_9 = data.loc[:,names[8]].values
    measured_values_10 = data.loc[:,names[9]].values
    measured_values_11= data.loc[:,names[10]].values
    measured_values_12 = data.loc[:,names[11]].values
    measured_values_13= data.loc[:,names[12]].values



    nbins=5
    # # 计算校准曲线
    # fraction_of_positives_1, mean_predicted_value_1 = calibration_curve(standard_values, measured_values_1, n_bins=nbins)
    # fraction_of_positives_2, mean_predicted_value_2 = calibration_curve(standard_values, measured_values_2, n_bins=nbins)
    # fraction_of_positives_3, mean_predicted_value_3 = calibration_curve(standard_values, measured_values_3, n_bins=nbins)
    # fraction_of_positives_4, mean_predicted_value_4 = calibration_curve(standard_values, measured_values_4, n_bins=nbins)
    # fraction_of_positives_5, mean_predicted_value_5 = calibration_curve(standard_values, measured_values_5, n_bins=nbins)
    # fraction_of_positives_6, mean_predicted_value_6 = calibration_curve(standard_values, measured_values_5, n_bins=nbins)
    # fraction_of_positives_7, mean_predicted_value_7 = calibration_curve(standard_values, measured_values_5, n_bins=nbins)
    # fraction_of_positives_8, mean_predicted_value_8 = calibration_curve(standard_values, measured_values_5, n_bins=nbins)
    # fraction_of_positives_9, mean_predicted_value_5 = calibration_curve(standard_values, measured_values_5, n_bins=nbins)
    # fraction_of_positives_10, mean_predicted_value_5 = calibration_curve(standard_values, measured_values_5, n_bins=nbins)
    # fraction_of_positives_11, mean_predicted_value_5 = calibration_curve(standard_values, measured_values_5, n_bins=nbins)
    # fraction_of_positives_12, mean_predicted_value_5 = calibration_curve(standard_values, measured_values_5, n_bins=nbins)
    # fraction_of_positives_13, mean_predicted_value_5 = calibration_curve(standard_values, measured_values_5, n_bins=nbins)


    # plt.figure(figsize=(8, 6))

    # # 绘制校准曲线
    # plt.plot(mean_predicted_value_1, fraction_of_positives_1, 's-', label=names[0])
    # plt.plot(mean_predicted_value_2, fraction_of_positives_2, 'o-', label=names[1])
    # plt.plot(mean_predicted_value_3, fraction_of_positives_3, '^-', label=names[2])
    # plt.plot(mean_predicted_value_4, fraction_of_positives_4, 'x-', label=names[3])
    # # plt.plot(mean_predicted_value_5, fraction_of_positives_5, '*-', label=names[4])
    #
    #
    # # 添加完美校准参考线
    # savepath="calibration"
    # os.makedirs(savepath,exist_ok=True)
    #
    # plt.plot([0, 1], [0, 1], '--', label='Perfectly Calibrated',color="gray")
    #
    # # 添加标签和标题
    # plt.xlabel('Mean Predicted Value')
    # plt.ylabel('Fraction of Positives')
    # plt.title('Calibration Curve')
    #
    # # 添加图例
    # plt.legend(loc='lower right')
    # plt.savefig(os.path.join(savepath,"calibration_two.tif"),format="tiff",dpi=600)
    # plt.savefig(os.path.join(savepath,"calibration_two.png"),dpi=600)
    # # 显示图形
    # plt.show()

    value_1 = brier_score_loss(standard_values, measured_values_1)
    value_2 = brier_score_loss(standard_values, measured_values_2)
    value_3 = brier_score_loss(standard_values, measured_values_3)
    value_4 = brier_score_loss(standard_values, measured_values_4)
    value_5 = brier_score_loss(standard_values, measured_values_5)
    value_6 = brier_score_loss(standard_values, measured_values_6)
    value_7 = brier_score_loss(standard_values, measured_values_7)
    value_8 = brier_score_loss(standard_values, measured_values_8)
    value_9 = brier_score_loss(standard_values, measured_values_9)
    value_10 = brier_score_loss(standard_values, measured_values_10)
    value_11 = brier_score_loss(standard_values, measured_values_11)
    value_12 = brier_score_loss(standard_values, measured_values_12)
    value_13 = brier_score_loss(standard_values, measured_values_13)

    print(value_1,value_2,value_3,value_4,value_5,value_6,value_7,value_8,value_9,value_10,value_11,value_12,value_13)


def plot_brief():
    data = pd.read_csv('concat.csv')

    # 选择标准值列
    standard_values = data['label'].values
    names=data.columns.tolist()[:4]
    # names = data.columns.tolist()[:]
    print(names)

    # 选择实际测量值列
    measured_values_1 = data.loc[:, names[0]].values
    measured_values_2 = data.loc[:, names[1]].values
    measured_values_3 = data.loc[:, names[2]].values
    measured_values_4 = data.loc[:, names[3]].values
    # measured_values_5 = data.loc[:, names[4]].values


    nbins = 5
    # # 计算校准曲线
    fraction_of_positives_1, mean_predicted_value_1 = calibration_curve(standard_values, measured_values_1, n_bins=nbins)
    fraction_of_positives_2, mean_predicted_value_2 = calibration_curve(standard_values, measured_values_2, n_bins=nbins)
    fraction_of_positives_3, mean_predicted_value_3 = calibration_curve(standard_values, measured_values_3, n_bins=nbins)
    fraction_of_positives_4, mean_predicted_value_4 = calibration_curve(standard_values, measured_values_4, n_bins=nbins)
    # fraction_of_positives_5, mean_predicted_value_5 = calibration_curve(standard_values, measured_values_5, n_bins=nbins)


    plt.figure(figsize=(8, 6))
    value_1 = brier_score_loss(standard_values, measured_values_1)
    value_2 = brier_score_loss(standard_values, measured_values_2)
    value_3 = brier_score_loss(standard_values, measured_values_3)
    value_4 = brier_score_loss(standard_values, measured_values_4)
    # value_5 = brier_score_loss(standard_values, measured_values_5)

    # # 绘制校准曲线
    plt.plot(mean_predicted_value_1, fraction_of_positives_1, 's-', label=names[0]+f" (Brier: {value_1:.4f})")
    plt.plot(mean_predicted_value_2, fraction_of_positives_2, 'o-', label=names[1]+f" (Brier: {value_2:.4f})")
    plt.plot(mean_predicted_value_3, fraction_of_positives_3, '^-', label=names[2]+f" (Brier: {value_3:.4f})")
    plt.plot(mean_predicted_value_4, fraction_of_positives_4, 'x-', label=names[3]+f" (Brier: {value_4:.4f})")
    # plt.plot(mean_predicted_value_5, fraction_of_positives_5, '*-', label=names[4]+f" (Brier: {value_5:.4f})")
    #
    #+f" (Brier: {value_4:.4f}"
    # 添加完美校准参考线
    savepath="calibration"
    os.makedirs(savepath,exist_ok=True)

    plt.plot([0, 1], [0, 1], '--', label='Perfectly Calibrated',color="gray")

    # 添加标签和标题
    plt.xlabel('Mean Predicted Value')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')

    # 添加图例
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(savepath,"calibration_single.tif"),format="tiff",dpi=600)
    plt.savefig(os.path.join(savepath,"calibration_single.png"),dpi=600)
    # 显示图形
    plt.show()





if __name__ == '__main__':
    plot_brief()