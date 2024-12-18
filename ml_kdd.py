
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,confusion_matrix, precision_score, recall_score, f1_score

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
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
tem = 0.02
bs = 128
seed = 5009
seed_round = 5
epochs = 800
epoch_online=1
sample_interval = 2784
flip_percent = 0.05

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

def evaluate(y,y_pred):
    # 混淆矩阵
    print("Confusion matrix")
    print(confusion_matrix(y, y_pred))
    # Accuracy 
    print('Accuracy ',accuracy_score(y, y_pred))
    # Precision 
    print('Precision ',precision_score(y, y_pred))
    # Recall
    print('Recall ',recall_score(y, y_pred))
    # F1 score
    print('F1 score ',f1_score(y,y_pred))
    
# 随机森林
rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)
rf_predictions = rf_model.predict(x_test)
print("Random Forest")
evaluate(y_test,rf_predictions)

# 支持向量机
svm_model = SVC()
svm_model.fit(x_train, y_train)
svm_predictions = svm_model.predict(x_test)
print("Support Vector Machine")
evaluate(y_test,svm_predictions)

# XGBoost
xgb_model = XGBClassifier()
xgb_model.fit(x_train, y_train)
xgb_predictions = xgb_model.predict(x_test)
print("XGBoost")
evaluate(y_test,xgb_predictions)

# 决策树
dt_model = DecisionTreeClassifier()
dt_model.fit(x_train, y_train)
dt_predictions = dt_model.predict(x_test)
print("Decision Tree")
evaluate(y_test,dt_predictions)
