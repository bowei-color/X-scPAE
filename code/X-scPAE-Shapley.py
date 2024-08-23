# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 18:21:36 2024

@author: Administrator
"""


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import shap

print(torch.__version__)

# 检查CUDA是否可用
if torch.cuda.is_available():
    print("CUDA is available. GPU is ready for use.")
else:
    print("CUDA is not available. Using CPU instead.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据
data = pd.read_csv("C:/D盘/Code/SpyderData/HelPredictor-main/data/E_MTAB_3929.csv")

X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 标准化特征
scaler = StandardScaler()
X = scaler.fit_transform(X)

# PCA处理函数
def PCA_deal(X):
    # 创建PCA对象，保留所有主成分以计算特征重要性
    pca = PCA(50)
    pca.fit(X)

    # 获取主成分的载荷矩阵
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    # 计算每个特征的重要性
    n = loadings.shape[1]
    sum_load = 0
    for i in range(loadings.shape[1]):
        sum_load += (i + 1)

    feature_importance = np.zeros(loadings.shape[0])
    for i in range(loadings.shape[0]):
        sum_abs_loadings = 0
        for j in range(loadings.shape[1]):
            sum_abs_loadings += ((n - j) / sum_load) * abs(loadings[i, j])
        feature_importance[i] = sum_abs_loadings

    # 根据贡献度排序特征
    feature_importance_df = pd.DataFrame({'feature': data.columns[1:], 'importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    # 计算前n个特征的重要性加和占到全部特征重要性加和的50%时该个数
    cumulative_importance = np.cumsum(feature_importance_df['importance'])
    total_importance = cumulative_importance.iloc[-1]
    n_features = np.searchsorted(cumulative_importance, 0.1 * total_importance) + 1

    # 选择前n个重要特征
    selected_features = feature_importance_df['feature'].head(n_features).tolist()

    # 过滤数据，只保留选择的重要特征  
    X = data[selected_features].values
    X = scaler.fit_transform(X)  # 对选择的特征重新标准化
    
    return X, selected_features

# 注意力机制定义
class FeatureAttention(nn.Module):
    def __init__(self, input_dim):
        super(FeatureAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.input_dim = input_dim

    def forward(self, x):
        # 保持输入的形状 (batch_size, input_dim)
        Q = self.query(x)  # (batch_size, input_dim)
        K = self.key(x)    # (batch_size, input_dim)
        V = self.value(x)  # (batch_size, input_dim)

        # 将 K 和 Q 进行矩阵乘法，得到注意力分数
        attention_scores = torch.matmul(Q.unsqueeze(2), K.unsqueeze(1)) / torch.sqrt(torch.tensor(self.input_dim, dtype=torch.float32))  # (batch_size, input_dim, input_dim)
        attention_weights = self.softmax(attention_scores)  # (batch_size, input_dim, input_dim)

        # 对 V 进行加权求和
        out = torch.matmul(attention_weights, V.unsqueeze(2)).squeeze(2)  # (batch_size, input_dim)
        return out, attention_weights

# 自编码器模型定义
class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_class):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.attention = FeatureAttention(hidden_size2)
        self.dropout = nn.Dropout(0.4)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size2, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, input_size),
            nn.Sigmoid(),
        )

        self.classifier = nn.Linear(hidden_size2, num_class)

    def forward(self, x):
        encoded = self.encoder(x)
        attended, attn_weights = self.attention(encoded)
        attended = self.dropout(attended)
        decoded = self.decoder(attended)
        logits = self.classifier(attended)  # 注意这里不需要平均池化，因为attended已经是(batch_size, hidden_size2)
        return logits

# 数据预处理和模型定义部分结束，以下是使用SHAP计算特征重要性的代码

X_pca, selected_features = PCA_deal(X)

X_pca.shape

# 模型参数
input_size = X_pca.shape[1]  # 特征的数量
hidden_size1 = 128  # 第一个隐藏层
hidden_size2 = 64  # 第二个隐藏层
num_classes = len(np.unique(y))

model = AutoEncoder(input_size, hidden_size1, hidden_size2, num_classes).to(device)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)




target_class = num_classes

feature_names = selected_features

# # 使用SHAP解释自编码器模型
def explain_model_with_shap(model, X_train, feature_names, target_class):

    # 将模型设为评估模式
    model.eval()
    
    # 定义SHAP解释器
    explainer = shap.DeepExplainer(model, torch.tensor(X_train, dtype=torch.float32).to(device))
    
    # 计算SHAP值
    shap_values = explainer.shap_values(torch.tensor(X_train, dtype=torch.float32).to(device))
    
    # 将所有类别的SHAP值进行平均
    shap_values_mean = np.mean(shap_values, axis=2)
    # 将所有类别的SHAP值进行平均
    # shap_values_mean1 = np.mean(shap_values, axis=2)
    
    # 计算所有样本特征重要性的平均值
    feature_importance = np.mean(shap_values_mean, axis=0)
    
    feature_importance.shape
    
    # 检查长度是否一致
    if len(feature_names) != len(feature_importance):
        raise ValueError("Feature names and importance scores must have the same length")
    
    # 创建包含特征名称和重要性的DataFrame
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
    
    # 根据重要性排序
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    
    # 选择重要性前20个特征
    top_20_features = feature_importance_df.head(20)
    

    return top_20_features['feature'].tolist(), feature_importance_df


shap_sorted_features, feature_importance_df = explain_model_with_shap(model, X_train, selected_features, num_classes)

