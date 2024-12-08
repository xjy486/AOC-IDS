# AOC-IDS😁

本项目基于  
https://github.com/xinchen930/AOC-IDS/

原始论文  
[AOC-IDS: Autonomous Online Framework with Contrastive Learning for Intrusion Detection" (Infocom 2024)
Xinchen Zhang, Running Zhao, Zhihan Jiang, Zhicong Sun, Yulong Ding, Edith C.H. Ngai, Shuang-hua Yang.](https://ieeexplore.ieee.org/document/10621346/)

我在作者公开的代码基础上，补充了其他缺失的一部分实验，包括数据预处理等环节。

**note:由于我无法完全保证与作者相同的实验环境，所以很多实验结果无法与论文中的结果保持一致。**

代码可能有许多有错误和疏忽的地方，还请多多指教。

# 文件说明 😘

## 数据集 🤡

### 原始数据集 🤡

KDD 数据集：NSL_KDD_Test.csv, NSL_KDD_Train.csv
NB15 数据集：UNSW_NB15_testing-set.csv, UNSW_NB15_training-set.csv

### 预处理后的数据集 🤡

KDD 数据集：en_KDDTrain+.csv, en_KDDTest+.csv
NB15 数据集：太大，没传上来（可以通过运行代码 preprocess.ipynb 获取）
**建议：使用原始代码仓库中提供的数据集，此处只是模拟了预处理的过程，无法保证结果与作者提供的预处理数据集完全一致。**

## 代码说明 🤡

1.preprocess.ipynb：预处理原始数据集，生成预处理后的数据集

2.nsl-kdd.ipynb：使用 NSL-KDD 数据集 训练，在线学习

3.unsw_nb15.ipynb：使用 NB15 数据集 训练，在线学习

4.消融实验\_infonce.ipynb: 消融实验，使用 InfoNCE Loss 替换 CRC loss

5.消融实验\_只使用编码器.ipynb: 消融实验，只使用编码器的预测结果

6.消融实验\_只使用解码器.ipynb: 消融实验，只使用解码器的预测结果

> 还要一些消融实验，因为不会写，没有做实验。🤡

# Notice❗

1. 由于我无法保证与作者相同的实验环境，所以很多实验结果无法与论文中的结果保持一致。
2. 代码可能有许多有错误和疏忽的地方，还请多多指教。
3. 原作者提供的代码，有指定 5 次训练轮次，我这里没有指定，默认只训练一次。

# 感谢和参考 ❤️❤️❤️

感谢论文作者公开源代码，方便其他人在此基础上进行研究。

其他参考的代码：
https://github.com/Mamcose/NSL-KDD-Network-Intrusion-Detection/tree/master  
https://github.com/thinline72/nsl-kdd
