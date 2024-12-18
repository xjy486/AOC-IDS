import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
unseen ="""
{
  "dos": {
    "AOC": 0.9994175888177053,
    "RF": 0.24985439720442632,
    "SVM": 0.30809551543389635,
    "XGB": 0.3302271403610949,
    "DT": 0.47874199184624344
  },
  "probe": {
    "AOC": 1.0,
    "RF": 0.46311787072243343,
    "SVM": 0.8365019011406845,
    "XGB": 0.4448669201520912,
    "DT": 0.5787072243346008
  },
  "r2l": {
    "AOC": 1.0,
    "RF": 0.0036036036036036037,
    "SVM": 0.009009009009009009,
    "XGB": 0.02882882882882883,
    "DT": 0.043243243243243246
  },
  "u2r": {
    "AOC": 1.0,
    "RF": 0.018404907975460124,
    "SVM": 0.04294478527607362,
    "XGB": 0.049079754601226995,
    "DT": 0.6441717791411042
  }
}
"""
seen = """
{
  "dos": {
    "AOC": 1.0,
    "RF": 0.9956453579515764,
    "SVM": 0.9515763804215294,
    "XGB": 0.9963421006793242,
    "DT": 0.9916390872670267
  },
  "probe": {
    "AOC": 1.0,
    "RF": 0.9954792043399638,
    "SVM": 0.8887884267631103,
    "XGB": 0.9502712477396021,
    "DT": 0.9014466546112115
  },
  "r2l": {
    "AOC": 1.0,
    "RF": 0.09731696225557071,
    "SVM": 0.04911323328785812,
    "XGB": 0.1341518872214643,
    "DT": 0.19054115507048658
  },
  "u2r": {
    "AOC": 1.0,
    "RF": 0.05405405405405406,
    "SVM": 0.05405405405405406,
    "XGB": 0.16216216216216217,
    "DT": 0.5675675675675675
  }
}
"""
# 美化图表
sns.set_style("whitegrid")
sns.set_context("talk")
sns.set_palette("Set2")

dict_seen = json.loads(seen)
dict_unseen = json.loads(unseen)

dict_seen = {outer_k: {inner_k: round(inner_v * 100, 4) for inner_k, inner_v in outer_v.items()} for outer_k, outer_v in dict_seen.items()}
dict_unseen = {outer_k: {inner_k: round(inner_v * 100, 4) for inner_k, inner_v in outer_v.items()} for outer_k, outer_v in dict_unseen.items()}

categories = ['dos', 'probe', 'r2l', 'u2r']
methods = ['AOC', 'RF', 'SVM', 'XGB', 'DT']

fig, axis = plt.subplots(1, 4, figsize=(14, 6))
for i, c in enumerate(categories):
    values = list(dict_unseen[c].values())
    seen_values = list(dict_seen[c].values())
    
    # 绘制柱状图
    barplot = sns.barplot(x=methods, y=values, ax=axis[i], palette="Set1")
    # axis[i].set_title(c, pad=20, fontsize=14, fontweight='bold')
    axis[i].set_ylabel('Detection Rate (%)', fontsize=12)
    axis[i].set_xlabel(c, fontsize=14,fontweight='bold')
    
    # 隐藏上方和右侧的边框
    if i != 0:
        axis[i].set_ylabel('')  # 移除其他子图的 y 轴标签
        axis[i].set_yticklabels([])  # 移除其他子图的 y 轴刻度标签
        axis[i].spines['top'].set_visible(False)
        axis[i].spines['right'].set_visible(False)
    else:
        axis[i].spines['top'].set_visible(False)
        axis[i].spines['right'].set_visible(False)
    
    axis[i].grid(False)  # 移除网格
    
    # 添加注释
    for p, seen_value in zip(barplot.patches, seen_values):
        barplot.annotate(f'{format(p.get_height(), ".2f")}', 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='center', 
                         xytext=(0, 15),  
                         textcoords='offset points',
                         fontsize=10,
                         color='red')  # 设置红色注释
        barplot.annotate(f'({format(seen_value, ".2f")})', 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='center', 
                         xytext=(0, 4),  
                         textcoords='offset points',
                         fontsize=10,
                         color='darkblue')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# fig.text(0.5, 0.03, '100: Detection Rate for Unseen Attacks', ha='center', fontsize=12, color='red')
# fig.text(0.5, 0.00, '100: Detection Rate for Seen Attacks', ha='center', fontsize=12,color='darkblue')
fig.text(0.78, 0.98, '100: Detection Rate for Unseen Attacks', ha='left', fontsize=12, color='red')
fig.text(0.78, 0.93, '100: Detection Rate for Seen Attacks', ha='left', fontsize=12, color='darkblue')

plt.show()
