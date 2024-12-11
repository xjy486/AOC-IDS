
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,confusion_matrix, precision_score, recall_score, f1_score
import json

# 假设 dict_seen 和 dict_unseen 已经定义并填充数据
from sklearn.metrics import accuracy_score
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')

def get_dataset():
    train_data=pd.read_csv('en_KDDTrain+.csv')
    test_data=pd.read_csv('en_KDDTest+.csv')
    y_train=train_data['label2']
    y_test=test_data['label2']
    X_train=train_data.drop(columns=['label2','class','label'])
    X_test=test_data.drop(columns=['label2','class','label'])
    normalize=MinMaxScaler()
    X_train=normalize.fit_transform(X_train)
    X_test=normalize.fit_transform(X_test)
    return X_train,y_train,X_test,y_test

x_train,y_train,x_test,y_test=get_dataset()
   
    
def predict(model,data):
    y = data['label2']
    x = data.drop(columns=['label','label2','class'])
    y_pred = model.predict(x)
     # 混淆矩阵
    print("Confusion matrix")
    print(confusion_matrix(y, y_pred))
     # Precision 
    print('Precision ',precision_score(y, y_pred))
    # Recall
    print('Recall ',recall_score(y, y_pred))
    # Accuracy 
    print('Accuracy ',accuracy_score(y, y_pred))
    # F1 score
    print('F1 score ',f1_score(y,y_pred))
    return recall_score(y,y_pred)

# 随机森林
rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)


# 支持向量机
svm_model = SVC()
svm_model.fit(x_train, y_train)


# XGBoost
xgb_model = XGBClassifier()
xgb_model.fit(x_train, y_train)


# 决策树
dt_model = DecisionTreeClassifier()
dt_model.fit(x_train, y_train)


###############################################################
df_train = pd.read_csv('en_KDDTrain+.csv')
df_test = pd.read_csv('en_KDDTest+.csv')
# 归一化
exclude_columns=['label','label2','class']
normalize_columns = [c for c in df_train.columns if c not in exclude_columns]

normalize = MinMaxScaler().fit(df_test[normalize_columns])
df_test[normalize_columns] = normalize.transform(df_test[normalize_columns])



attacks_train = set(df_train['label'].unique())
attacks_test = set(df_test['label'].unique())
# 仅仅在测试集中出现的攻击类别
attacks_only_test = attacks_test - attacks_train
indices_unseen = df_test[df_test['label'].isin(attacks_only_test)].index
# 在训练集和测试集中都出现的攻击类别
indices_seen = df_test[df_test['label'].isin(attacks_train&attacks_test)].index

df_seen = df_test.loc[indices_seen]
df_unseen = df_test.loc[indices_unseen]
# seen
seen_dos = df_seen[df_seen['class']=='DoS']
seen_probe = df_seen[df_seen['class']=='Probe']
seen_r2l = df_seen[df_seen['class']=='R2L']
seen_u2r = df_seen[df_seen['class']=='U2R']
# unseen
unseen_dos = df_unseen[df_unseen['class']=='DoS']
unseen_probe = df_unseen[df_unseen['class']=='Probe']
unseen_r2l = df_unseen[df_unseen['class']=='R2L']
unseen_u2r = df_unseen[df_unseen['class']=='U2R']

dict_seen ={
    "dos":{"RF":0.0,"SVM":0.0,"XGB":0.0,"DT":0.0},
    "probe":{"RF":0.0,"SVM":0.0,"XGB":0.0,"DT":0.0},
    "r2l":{"RF":0.0,"SVM":0.0,"XGB":0.0,"DT":0.0},
    "u2r":{"RF":0.0,"SVM":0.0,"XGB":0.0,"DT":0.0}
}
dict_unseen ={
    "dos":{"RF":0.0,"SVM":0.0,"XGB":0.0,"DT":0.0},
    "probe":{"RF":0.0,"SVM":0.0,"XGB":0.0,"DT":0.0},
    "r2l":{"RF":0.0,"SVM":0.0,"XGB":0.0,"DT":0.0},
    "u2r":{"RF":0.0,"SVM":0.0,"XGB":0.0,"DT":0.0}
}

print("RF dos seen:")
dict_seen["dos"]["RF"]=predict(rf_model,seen_dos)
print("RF dos unseen:")
dict_unseen["dos"]["RF"]=predict(rf_model,unseen_dos)
print("RF probe seen:")
dict_seen["probe"]["RF"]=predict(rf_model,seen_probe)
print("RF probe unseen:")
dict_unseen["probe"]["RF"]=predict(rf_model,unseen_probe)
print("RF r2l seen:")
dict_seen["r2l"]["RF"]=predict(rf_model,seen_r2l)
print("RF r2l unseen:")
dict_unseen["r2l"]["RF"]=predict(rf_model,unseen_r2l)
print("RF u2r seen:")
dict_seen["u2r"]["RF"]=predict(rf_model,seen_u2r)
print("RF u2r unseen:")
dict_unseen["u2r"]["RF"]=predict(rf_model,unseen_u2r)
print("SVM dos seen:")
dict_seen["dos"]["SVM"]=predict(svm_model,seen_dos)
print("SVM dos unseen:")
dict_unseen["dos"]["SVM"]=predict(svm_model,unseen_dos)
print("SVM probe seen:")
dict_seen["probe"]["SVM"]=predict(svm_model,seen_probe)
print("SVM probe unseen:")
dict_unseen["probe"]["SVM"]=predict(svm_model,unseen_probe)
print("SVM r2l seen:")
dict_seen["r2l"]["SVM"]=predict(svm_model,seen_r2l)
print("SVM r2l unseen:")
dict_unseen["r2l"]["SVM"]=predict(svm_model,unseen_r2l)
print("SVM u2r seen:")
dict_seen["u2r"]["SVM"]=predict(svm_model,seen_u2r)
print("SVM u2r unseen:")
dict_unseen["u2r"]["SVM"]=predict(svm_model,unseen_u2r)
print("XGB dos seen:")
dict_seen["dos"]["XGB"]=predict(xgb_model,seen_dos)
print("XGB dos unseen:")
dict_unseen["dos"]["XGB"]=predict(xgb_model,unseen_dos)
print("XGB probe seen:")
dict_seen["probe"]["XGB"]=predict(xgb_model,seen_probe)
print("XGB probe unseen:")
dict_unseen["probe"]["XGB"]=predict(xgb_model,unseen_probe)
print("XGB r2l seen:")
dict_seen["r2l"]["XGB"]=predict(xgb_model,seen_r2l)
print("XGB r2l unseen:")
dict_unseen["r2l"]["XGB"]=predict(xgb_model,unseen_r2l)
print("XGB u2r seen:")
dict_seen["u2r"]["XGB"]=predict(xgb_model,seen_u2r)
print("XGB u2r unseen:")
dict_unseen["u2r"]["XGB"]=predict(xgb_model,unseen_u2r)
print("DT dos seen:")
dict_seen["dos"]["DT"]=predict(dt_model,seen_dos)
print("DT dos unseen:")
dict_unseen["dos"]["DT"]=predict(dt_model,unseen_dos)
print("DT probe seen:")
dict_seen["probe"]["DT"]=predict(dt_model,seen_probe)
print("DT probe unseen:")
dict_unseen["probe"]["DT"]=predict(dt_model,unseen_probe)
print("DT r2l seen:")
dict_seen["r2l"]["DT"]=predict(dt_model,seen_r2l)
print("DT r2l unseen:")
dict_unseen["r2l"]["DT"]=predict(dt_model,unseen_r2l)
print("DT u2r seen:")
dict_seen["u2r"]["DT"]=predict(dt_model,seen_u2r)
print("DT u2r unseen:")
dict_unseen["u2r"]["DT"]=predict(dt_model,unseen_u2r)



# 将 dict_seen 保存为 seen.json 文件
with open('seen.json', 'w') as seen_file:
    json.dump(dict_seen, seen_file, indent=4)

# 将 dict_unseen 保存为 unseen.json 文件
with open('unseen.json', 'w') as unseen_file:
    json.dump(dict_unseen, unseen_file, indent=4)

print("字典已保存为文件。")

