# -*- coding: utf-8 -*-



import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from torch.nn import functional as F
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from torch.autograd import Variable
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.stats import pearsonr, spearmanr
import argparse

print(torch.__version__)

# 检查CUDA是否可用
if torch.cuda.is_available():
    print("CUDA is available. GPU is ready for use.")
else:
    print("CUDA is not available. Using CPU instead.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#读取数据
def read_data(file_path):

    # 加载数据
    data = pd.read_csv(file_path)
    
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values
    
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    # 标准化特征
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y, data





#PCA处理函数
def PCA_deal(X, data):
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
    # 标准化特征
    scaler = StandardScaler()
    X = data[selected_features].values
    X = scaler.fit_transform(X)  # 对选择的特征重新标准化
    
    
    return X, selected_features, data[selected_features]




#定义注意力机制
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

#定义自编码器模型
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
        return encoded, attended, decoded, logits, attn_weights




#训练函数
def train(model, train_loader, reconstruction_criterion, classification_criterion, optimizer, reconstruction_weight):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0
    total_reconstruction_loss = 0.0
    total_classification_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        _, _, decoded, logits, _ = model(inputs)
        reconstruction_loss = reconstruction_criterion(decoded, inputs)
        classification_loss = classification_criterion(logits, labels)

        loss = reconstruction_weight * reconstruction_loss + (1 - reconstruction_weight) * classification_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_reconstruction_loss += reconstruction_loss.item()
        total_classification_loss += classification_loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    avg_reconstruction_loss = total_reconstruction_loss / len(train_loader)
    avg_classification_loss = total_classification_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy, avg_reconstruction_loss, avg_classification_loss

# 测试函数
def test(model, loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, _, _, logits, _ = model(inputs)
            _, predicted = torch.max(logits, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    return accuracy, precision, recall, f1

# 五折交叉验证
def run_cross_validation(model, X, y, num_classes, epochs, n_splits=5,  batch_size=64, lr=0.001, reconstruction_weight=0.5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    best_model_state = None
    best_accuracy = 0
    final_train_loader = None

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long())
        test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long())

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        reconstruction_criterion = nn.MSELoss()
        classification_criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), lr, weight_decay=0.0001)

        for epoch in range(epochs):
            train_loss, train_accuracy, reconstruction_loss, classification_loss = train(model, train_loader, reconstruction_criterion, classification_criterion, optimizer, reconstruction_weight)
            print(f"Epoch[{epoch+1}/{epochs}], Reconstruction Loss: {reconstruction_loss:.4f}, Classification Loss: {classification_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        accuracy, precision, recall, f1 = test(model, test_loader)
        results['accuracy'].append(accuracy)
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['f1'].append(f1)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state = model.state_dict()
            final_train_loader = train_loader

    model.load_state_dict(best_model_state)
    return results, final_train_loader






#性能评估函数
def evaluate(model, X_train, y_train, X_test, y_test, num_classes, epochs):
    
    # 运行五折交叉验证
    cv_results, final_train_loader = run_cross_validation(model, X_train, y_train, num_classes, epochs)
    for metric, scores in cv_results.items():
        # print(f"Cross-Validation {metric.capitalize()}: {scores}")
        print(f"Mean {metric.capitalize()}: {np.mean(scores):.4f}")

    # 在测试集上评估模型
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    test_accuracy, test_precision, test_recall, test_f1 = test(model, test_loader)
    
   
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")
    return test_accuracy, test_precision, test_recall, test_f1



    
    
    
    # 混淆矩阵函数
def Confusion_Matrix(model, X_test, y_test):
    
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 获取预测结果和实际标签
    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, _, _, logits, _ = model(inputs)
            _, predicted = torch.max(logits, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    return conf_matrix
    
    








# 注意力权重函数
def Attention_average(model, X_train):

    # 将训练数据转换为张量
    X_train_tensor = torch.tensor(X_train).float().to(device)
    _, _, _, _, attn_weights = model(X_train_tensor)
    
    # 将注意力权重矩阵转换为NumPy数组
    attn_weights_np = attn_weights.detach().cpu().numpy()
    
    # 计算注意力平均值
    attn_weights_np_avg = np.mean(attn_weights_np, axis=0)
    
    # 对attn_weights_np_avg进行对称处理
    n = attn_weights_np_avg.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            avg_val = (attn_weights_np_avg[i, j] + attn_weights_np_avg[j, i]) / 2
            attn_weights_np_avg[i, j] = avg_val
            attn_weights_np_avg[j, i] = avg_val
    
    # 提取最大的十个值及其下标
    flat_indices = np.argsort(attn_weights_np_avg, axis=None)[-10:]
    row_indices, col_indices = np.unravel_index(flat_indices, attn_weights_np_avg.shape)
    
    # 确保选择前10个唯一的行和列
    unique_indices = np.unique(np.concatenate((row_indices, col_indices)))[:10]
    
    # 如果唯一行和列少于10个，则添加更多行和列直到达到10个
    if len(unique_indices) < 10:
        additional_indices = np.setdiff1d(np.arange(n), unique_indices)[:10-len(unique_indices)]
        unique_indices = np.concatenate((unique_indices, additional_indices))
    

    
    # 按下标顺序排列
    sorted_indices = np.sort(unique_indices)
    sorted_top_attn_weights_np_avg = attn_weights_np_avg[np.ix_(sorted_indices, sorted_indices)]
    
    
    df = pd.DataFrame(sorted_top_attn_weights_np_avg)
    
    return df, unique_indices







#积分梯度函数
def integrated_gradients(model, x, target_class, baseline=None, steps=50):
    if baseline is None:
        baseline = torch.zeros_like(x)
        
      # 梯度累加
    scaled_inputs = [baseline + (float(i) / steps) * (x - baseline) for i in range(0, steps + 1)]
    classifier_grads = []
     
    for scaled_input in scaled_inputs:
        scaled_input = Variable(scaled_input, requires_grad=True)
        encoded, attended, decoded, logits, _ = model(scaled_input)
        model.zero_grad()
        
        # 计算分类器的梯度
        logit_target = logits[0, target_class]
        logit_target.backward(retain_graph=True)
        classifier_grads.append(scaled_input.grad.cpu().data.numpy()) 

    avg_classifier_grads = np.average(classifier_grads[:-1], axis=0)
    integrated_classifier_grad = (x.cpu().data.numpy() - baseline.cpu().data.numpy()) * avg_classifier_grads
    
    return integrated_classifier_grad


#计算积分梯度函数
def get_integrated_gradients(model, X_train, y_train, selected_features):
    all_classifier_grads, all_labels = [], []
    
    
    for i in range(len(X_train)):
        x = torch.tensor(X_train[i:i+1], requires_grad=True, dtype=torch.float32).to(device)
        target_class = y_train[i]
        classifier_grad = integrated_gradients(model, x, target_class)
        all_classifier_grads.append(classifier_grad)
        all_labels.append(target_class)
    
    all_classifier_grads = np.array(all_classifier_grads).squeeze()
    all_classifier_grads = np.abs(all_classifier_grads)
    
    
    
    # 创建一个DataFrame来保存特征归因值
    attributions = pd.DataFrame({
        'feature': selected_features,
        'classifier_attribution': np.abs(all_classifier_grads.mean(axis=0))
    })
    

    return attributions, all_classifier_grads, all_labels





# 积分梯度排序
def sort_clssifier_grads(clssifier_attributions,all_classifier_grads, topn):
    classifier_attribution = clssifier_attributions.sort_values(by='classifier_attribution', ascending=False)
    
    
    return classifier_attribution['feature'].head(20), classifier_attribution['feature'].head(topn)










# 积分梯度排序重要基因在各分类上的积分梯度函数
def Heatmap_of_selected_average_Grads(all_classifier_grads, selected_features, all_labels, attributions):
    
    # 标签映射
    label_mapping = {
        0: 'E5_EPI',
        1: 'E5_TE',
        2: 'E5_PE',
        3: 'E6_EPI',
        4: 'E6_TE',
        5: 'E6_PE',
        6: 'E7_EPI',
        7: 'E7_TE',
        8: 'E7_PE'
    }
    # 创建一个包含所有归因值和标签的DataFrame
    all_class_label = pd.DataFrame(all_classifier_grads, columns=selected_features)
    all_class_label['label'] = all_labels

    # 替换标签
    all_class_label['label'] = all_class_label['label'].map(label_mapping)
    
    # 按类别计算每个特征的平均归因值
    average_attributions_by_class = all_class_label.groupby('label').mean()
    
    classifier_attribution = attributions.sort_values(by='classifier_attribution', ascending=False)
    sort_genes = classifier_attribution['feature'].head(20)

    # 从average_attributions_by_class中选出对应的特征
    selected_average_attributions = average_attributions_by_class[sort_genes]
    
    
    return  selected_average_attributions




#计算不同类别上的积分梯度比例和皮尔森系数的函数
def Barplot_of_top10_Gradients(all_classifier_grads, selected_features, all_labels, attributions):
    # 标签映射
    label_mapping = {
        0: 'E5_EPI',
        1: 'E5_TE',
        2: 'E5_PE',
        3: 'E6_EPI',
        4: 'E6_TE',
        5: 'E6_PE',
        6: 'E7_EPI',
        7: 'E7_TE',
        8: 'E7_PE'
    }
    
    
    # 创建一个包含所有归因值和标签的DataFrame
    all_class_label = pd.DataFrame(all_classifier_grads, columns=selected_features)
    all_class_label['label'] = all_labels

    # 替换标签
    all_class_label['label'] = all_class_label['label'].map(label_mapping)
    
    # 按类别计算每个特征的平均归因值
    average_attributions_by_class = all_class_label.groupby('label').mean()
    
    # 计算前10个特征的平均归因值并排序
    classifier_attribution = attributions.sort_values(by='classifier_attribution', ascending=False)
    top10_genes = classifier_attribution['feature'].head(10)

    # 从average_attributions_by_class中选出对应的特征
    selected_average_attributions = average_attributions_by_class[top10_genes]
    
    # 计算每个特征在不同类别的积分梯度占比并转换为百分比
    proportions = selected_average_attributions.div(selected_average_attributions.sum(axis=1), axis=0) * 100
    
    # 标签分组
    groups = [['E5_EPI', 'E5_TE', 'E5_PE'], ['E6_EPI', 'E6_TE', 'E6_PE'], ['E7_EPI', 'E7_TE', 'E7_PE']]
    
    pearson_results = []

    # 计算皮尔森系数
    for group in groups:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                label1, label2 = group[i], group[j]
                vec1, vec2 = proportions.loc[label1], proportions.loc[label2]
                pearson_corr, _ = pearsonr(vec1, vec2)
                pearson_results.append({'Group': f"{label1} vs {label2}", 'Pearson Correlation': pearson_corr})

    # 保存皮尔森系数结果
    pearson_df = pd.DataFrame(pearson_results)

    return proportions, pearson_df











# CV2方法选择前20个特征
def CV2_features(data, y):
    X = data.iloc[:, 2:]
    cv2_scores = X.var() / X.mean()**2
    top_features = cv2_scores.nlargest(20).index
    return top_features

# PCA方法选择前20个特征
def PCA_features(data, y):
    # 创建PCA对象，保留所有主成分以计算特征重要性
    
    X = data.iloc[:, 2:].values


    # 标准化特征
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    pca = PCA(30)
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
    feature_importance_df = pd.DataFrame({'feature': data.columns[2:], 'importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    # 选择前n个重要特征
    selected_features = feature_importance_df['feature'].head(20).tolist()

    # 过滤数据，只保留选择的重要特征  
    X = data[selected_features].values
    X = scaler.fit_transform(X)  # 对选择的特征重新标准化
    
    return selected_features

    
# 随机选择20个特征的方法
def random_features(data):
    feature_names = data.columns[2:]  # 假设前两列不是特征列
    random_features = np.random.choice(feature_names, 20, replace=False)
    return random_features


# 用F-Score方法选择重要基因
def F_score(data, y):
    # 提取特征数据
    X = data.iloc[:, 2:].values
    
    # 使用SelectKBest和f_classif计算F-score
    selector = SelectKBest(score_func=f_classif, k=1000)  # 选择前20个特征
    X_new = selector.fit_transform(X, y)
    
    # 获取每个特征的F-score和p-value
    scores = selector.scores_
    p_values = selector.pvalues_
    
    # 生成特征名称（使用特征索引）
    feature_names = data.columns[2:]  # 假设前两列不是特征列
    selected_features = feature_names[selector.get_support()]
    
    # 将结果转换为DataFrame便于查看
    result = pd.DataFrame({'Feature': feature_names, 'F-Score': scores, 'p-value': p_values})
    
    # 按F-score排序并选择前20个特征
    top_features = result.nlargest(20, 'F-Score')['Feature']
    
    # 返回前20个特征名称
    return top_features

# 用重要基因做逻辑回归的函数
def logistic_regression(data, y, features):
    # 特征排序并获取值
    X = data[features].values
    
    # 对选择的特征重新标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 训练逻辑回归模型
    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train, y_train)
    
    # 模型预测
    y_pred_train = logistic_model.predict(X_train)
    y_pred_test = logistic_model.predict(X_test)
    
    # 计算评估指标
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_precision = precision_score(y_test, y_pred_test, average='macro')
    test_recall = recall_score(y_test, y_pred_test, average='macro')
    test_f1 = f1_score(y_test, y_pred_test, average='macro')
    return test_accuracy, test_precision, test_recall, test_f1

# 用重要基因做逻辑回归的函数
def logistic_regression_pearsonr_spearmanr(data, y, features):
    # 特征排序并获取值
    X = data[features].values
    
    # 对选择的特征重新标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 训练逻辑回归模型
    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train, y_train)
    
    # 模型预测
    y_pred_train = logistic_model.predict(X_train)
    y_pred_test = logistic_model.predict(X_test)
    
    # 计算评估指标
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_precision = precision_score(y_test, y_pred_test, average='macro')
    test_recall = recall_score(y_test, y_pred_test, average='macro')
    test_f1 = f1_score(y_test, y_pred_test, average='macro')
    
    # 计算皮尔森和斯皮尔曼系数
    pearson_corr, _ = pearsonr(y_test, y_pred_test)
    spearman_corr, _ = spearmanr(y_test, y_pred_test)
    
    return test_accuracy, test_precision, test_recall, test_f1, pearson_corr, spearman_corr
    
# 各方法使用重要基因做分类预测的性能对比函数
def Different_methods_performance_comparison_importance_genes(data, y, sorted_features):
    results = []
    

    # X-scPAE方法选择的前20个基因做逻辑回归
    XscPAE_test_accuracy, XscPAE_test_precision, XscPAE_test_recall, XscPAE_test_f1,pearson_corr, spearman_corr = logistic_regression_pearsonr_spearmanr(data, y, sorted_features)
    results.append(["X-scPAE", XscPAE_test_accuracy, XscPAE_test_precision, XscPAE_test_recall, XscPAE_test_f1])
    
    
    # 用F-Score方法选择的前20个基因做逻辑回归
    features = F_score(data, y)
    FS_test_accuracy, FS_test_precision, FS_test_recall, FS_test_f1 = logistic_regression(data, y, features)
    results.append(["F-score", FS_test_accuracy, FS_test_precision, FS_test_recall, FS_test_f1])
    
    # 随机选择20个基因做逻辑回归
    random_features_list = random_features(data)
    random_test_accuracy, random_test_precision, random_test_recall, random_test_f1 = logistic_regression(data, y, random_features_list)
    results.append(["Random", random_test_accuracy, random_test_precision, random_test_recall, random_test_f1])
    
    # 用CV2方法选择的前20个基因做逻辑回归
    cv2_features_list = CV2_features(data, y)
    CV2_test_accuracy, CV2_test_precision, CV2_test_recall, CV2_test_f1 = logistic_regression(data, y, cv2_features_list)
    results.append(["CV2", CV2_test_accuracy, CV2_test_precision, CV2_test_recall, CV2_test_f1])
    
    # 用PCA方法选择的前20个特征做逻辑回归
    pca_features_list = PCA_features(data, y)
    PCA_test_accuracy, PCA_test_precision, PCA_test_recall, PCA_test_f1 = logistic_regression(data, y, pca_features_list)
    results.append(["PCA", PCA_test_accuracy, PCA_test_precision, PCA_test_recall, PCA_test_f1])
    
    comparison_df = pd.DataFrame(results, columns=["Feature selection method", "Accuracy", "Precision", "Recall", "F1"])
    
    # Prepare data for plotting
    melted_df = comparison_df.melt(id_vars=["Feature selection method"], value_vars=["Accuracy", "Precision", "Recall", "F1"], var_name="Evaluation indicators", value_name="Score")
    
   
    
    return melted_df






# 二分类用重要基因做逻辑回归的函数
def logistic_regression_binary_classification(data, y, features):
    # 特征排序并获取值
    X = data[features].values
    
    # 对选择的特征重新标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 训练逻辑回归模型
    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train, y_train)
    
    # 模型预测概率
    y_pred_prob = logistic_model.predict_proba(X_test)[:, 1]
    
    # 计算AUROC
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    
    return roc_auc, y_test, y_pred_prob


# 二分类各方法使用重要基因做分类预测的性能对比
def Different_methods_performance_comparison_importance_genes_binary_classification(data, y, sorted_features):
    unique_classes = np.unique(y)
    
    results = []
    
    # 只选择一次特征
    XscPAE_features = sorted_features
    F_score_features = F_score(data, y)
    random_features_list = random_features(data)
    CV2_features_list = CV2_features(data, y)
    PCA_features_list = PCA_features(data, y)

    for cls in unique_classes:
        binary_y = np.where(y == cls, 1, 0)
        
        

        # X-scPAE方法选择的前20个基因做逻辑回归
        XscPAE_roc_auc, y_test, y_pred_prob = logistic_regression_binary_classification(data, binary_y, XscPAE_features)
        
        # 用F-Score方法选择的前20个基因做逻辑回归
        FS_roc_auc, y_test, y_pred_prob = logistic_regression_binary_classification(data, binary_y, F_score_features)
        
        # 随机选择20个基因做逻辑回归
        random_roc_auc, y_test, y_pred_prob = logistic_regression_binary_classification(data, binary_y, random_features_list)
        
        # 用CV2方法选择的前20个基因做逻辑回归
        CV2_roc_auc, y_test, y_pred_prob = logistic_regression_binary_classification(data, binary_y, CV2_features_list)

        
        # 用PCA方法选择的前20个特征做逻辑回归
        PCA_roc_auc, y_test, y_pred_prob = logistic_regression_binary_classification(data, binary_y, PCA_features_list)
        
        
        
        # 设置字体大小
        plt.rcParams.update({'font.size': 20})  # 这里设置全局字体大小为15，你可以根据需要调整
        
        if cls == 3:
            results.append(['X-scPAE','E6_EPI', XscPAE_roc_auc, y_test, y_pred_prob])
            results.append(['F-Score','E6_EPI', FS_roc_auc, y_test, y_pred_prob])
            results.append(['Random','E6_EPI', random_roc_auc, y_test, y_pred_prob])
            results.append(['CV2','E6_EPI',  CV2_roc_auc, y_test, y_pred_prob])
            results.append(['PCA','E6_EPI', PCA_roc_auc, y_test, y_pred_prob])
            
        elif cls==4:
            results.append(['X-scPAE','E6_TE', XscPAE_roc_auc, y_test, y_pred_prob])
            results.append(['F-Score','E6_TE', FS_roc_auc, y_test, y_pred_prob])
            results.append(['Random','E6_TE', random_roc_auc, y_test, y_pred_prob])
            results.append(['CV2','E6_TE',  CV2_roc_auc, y_test, y_pred_prob])
            results.append(['PCA','E6_TE', PCA_roc_auc, y_test, y_pred_prob])
            
        elif cls==5:
            results.append(['X-scPAE','E6_PE', XscPAE_roc_auc, y_test, y_pred_prob])
            results.append(['F-Score','E6_PE', FS_roc_auc, y_test, y_pred_prob])
            results.append(['Random','E6_PE', random_roc_auc, y_test, y_pred_prob])
            results.append(['CV2','E6_PE',  CV2_roc_auc, y_test, y_pred_prob])
            results.append(['PCA','E6_PE', PCA_roc_auc, y_test, y_pred_prob])
            
    comparison_df = pd.DataFrame(results, columns=["Feature selection method", "lineage_stage","auc", "test", "pred_prob"])
      
      # Prepare data for plotting
    melted_df = comparison_df.melt(id_vars=["Feature selection method"], value_vars=["lineage_stage", "auc", "test", "pred_prob"], var_name="Evaluation indicators", value_name="Score")
    
    return melted_df






#比较重要基因在不同类别中的表达水平函数
def expression_different_categories_of_important_genes(attributions, data, y):
    
    
    
    # 标签映射
    label_mapping = {0: 'I5', 1: 'T5', 2: 'P5', 3: 'I6', 4: 'T6',  5: 'P6' , 6: 'I7', 7: 'T7', 8: 'P7'}
    
    # 按归因值排序
    attributions = attributions.sort_values(by='classifier_attribution', ascending=False)
    
    # 统计分析：比较重要基因在不同类别中的表达水平
    important_genes = attributions['feature'].head(10).values
    important_genes_data = data[important_genes]

    # 将类别信息添加到数据中
    # important_genes_data['label'] = y
    
    # 将 y 转换为 pandas Series 并进行标签映射
    y_mapped = pd.Series(y).map(label_mapping)
    
    # 将类别信息添加到数据中
    important_genes_data['label'] = pd.Categorical(y_mapped, categories=['I5', 'T5', 'P5','I6', 'T6',  'P6' ,'I7', 'T7', 'P7'], ordered=True)
    
    # 标准化数据（不包括label列）
    scaler = StandardScaler()
    important_genes_data[important_genes] = scaler.fit_transform(important_genes_data[important_genes])
    
    
    return important_genes_data, important_genes








#根据重要基因进行PCA的函数
def PCA_cluster(data, sorted_features, labels):
    # 使用选定的特征进行降维
    selected_features = data[sorted_features]

    # 数据标准化
    scaler = StandardScaler()
    selected_features_scaled = scaler.fit_transform(selected_features)

    # 进行PCA降维
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(selected_features_scaled)

    # 将降维后的特征和标签组合成一个DataFrame
    reduced_df = pd.DataFrame(reduced_features, columns=['PCA Feature 1', 'PCA Feature 2'])
    reduced_df['Original Label'] = labels

    # 根据标签分组
    def label_groupdevelopmental_stage(label):
        if label in [0, 1, 2]:
            return 'E5'
        elif label in [3, 4, 5]:
            return 'E6'
        else:
            return 'E7'

    def label_inferred_lineag(label):
        if label in [0, 3, 6]:
            return 'EPI'
        elif label in [1, 4, 7]:
            return 'TE'
        else:
            return 'PE'

    def label_stage_lineag(label):
        if label == 0:
            return 'E5-EPI'
        elif label == 1:
            return 'E5-TE'
        elif label == 2:
            return 'E5-PE'
        elif label == 3:
            return 'E6-EPI'
        elif label == 4:
            return 'E6-TE'
        elif label == 5:
            return 'E6-PE'
        elif label == 6:
            return 'E7-EPI'
        elif label == 7:
            return 'E7-TE'
        else:
            return 'E7-PE'

    reduced_df['groupdevelopmental_stage'] = reduced_df['Original Label'].apply(label_groupdevelopmental_stage)
    reduced_df['inferred_lineag'] = reduced_df['Original Label'].apply(label_inferred_lineag)
    reduced_df['Mapped Label'] = reduced_df['Original Label'].apply(label_stage_lineag)
    
    return reduced_df



    





#根据特征进行PCA的函数
def PCA_cluster_feature(sorted_features, labels, type):
    # 使用选定的特征进行降维
    selected_features = sorted_features

    # 数据标准化
    scaler = StandardScaler()
    selected_features_scaled = scaler.fit_transform(selected_features)

    # 进行PCA降维
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(selected_features_scaled)

    # 将降维后的特征和标签组合成一个DataFrame
    reduced_df = pd.DataFrame(reduced_features, columns=['PCA Feature 1', 'PCA Feature 2'])
    reduced_df['Original Label'] = labels
    
    
    # 根据标签分组
    def label_groupdevelopmental_stage(label):
        if label in [0, 1, 2]:
            return 'E5'
        elif label in [3, 4, 5]:
            return 'E6'
        else:
            return 'E7'

    def label_inferred_lineag(label):
        if label in [0, 3, 6]:
            return 'EPI'
        elif label in [1, 4, 7]:
            return 'TE'
        else:
            return 'PE'

    def label_stage_lineag(label):
        if label == 0:
            return 'E5-EPI'
        elif label == 1:
            return 'E5-TE'
        elif label == 2:
            return 'E5-PE'
        elif label == 3:
            return 'E6-EPI'
        elif label == 4:
            return 'E6-TE'
        elif label == 5:
            return 'E6-PE'
        elif label == 6:
            return 'E7-EPI'
        elif label == 7:
            return 'E7-TE'
        else:
            return 'E7-PE'

    reduced_df['groupdevelopmental_stage'] = reduced_df['Original Label'].apply(label_groupdevelopmental_stage)
    reduced_df['inferred_lineag'] = reduced_df['Original Label'].apply(label_inferred_lineag)
    reduced_df['Mapped Label'] = reduced_df['Original Label'].apply(label_stage_lineag)

    
    return reduced_df




# 训练好的模型获取潜向量函数
def get_latent_vectors(model, X):
    model.eval()
    X_tensor = torch.tensor(X).float().to(device)
    with torch.no_grad():
        encoded, _, _, _, _ = model(X_tensor)
    
    return encoded.cpu().numpy()


# 训练好的模型获取attended向量函数
def get_attended_vectors(model, X):
    model.eval()
    X_tensor = torch.tensor(X).float().to(device)
    with torch.no_grad():
        _, attended, _, _, _ = model(X_tensor)
    
    return attended.cpu().numpy()





if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', type=int, default= 1000)
    parser.add_argument('--file_path', type=str, default="../data/E_MTAB_3929_data.csv")
    
    args = parser.parse_args()
    
    


    X, y, data = read_data(args.file_path)

    X_pca, selected_features, pca_selected_feature=PCA_deal(X, data)

    # 模型参数
    input_size = X_pca.shape[1]  # 特征的数量
    hidden_size1 = 128  # 第一个隐藏层
    hidden_size2 = 64  # 第二个隐藏层
    num_classes = len(np.unique(y))

    model = AutoEncoder(input_size, hidden_size1, hidden_size2, num_classes).to(device)


    #划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)
    
    
    
    
    #性能评估
    X_scPAE_accuracy, X_scPAE_precision, X_scPAE_recall, X_scPAE_f1 = evaluate(model, X_train, y_train, X_test, y_test, num_classes, args.epochs)

    
    
    #计算混淆矩阵
    conf_matrix = Confusion_Matrix(model, X_test, y_test)
    
    # 计算平均注意力
    attention_average_weights, unique_indices = Attention_average(model, X_train)
    
    
   
    
    
    #计算分类积分梯度
    clssifier_attributions, all_classifier_grads, all_labels = get_integrated_gradients(model, X_train, y_train, selected_features)
    
    topn = 76
    
    #梯度排序
    sorted_features, sorted_features_topn = sort_clssifier_grads(clssifier_attributions, all_classifier_grads, topn)
    
    
    #不同类别上的积分梯度
    selected_average_attributions = Heatmap_of_selected_average_Grads(all_classifier_grads, selected_features, all_labels, clssifier_attributions)
    
    #计算不同类别上的积分梯度比例和皮尔森系数
    top10_Gradients_groups, pearson_df = Barplot_of_top10_Gradients(all_classifier_grads, selected_features, all_labels, clssifier_attributions)
    
    #计算各方法使用重要基因做分类预测的性能
    melted_df = Different_methods_performance_comparison_importance_genes(data, y, sorted_features)
    
    
    #计算二分类时各方法使用重要基因做分类预测的性能
    binary_melted_df = Different_methods_performance_comparison_importance_genes_binary_classification(data, y, sorted_features)
    
    #重要基因在不同类别中的表达水平
    important_genes_data, important_genes = expression_different_categories_of_important_genes(clssifier_attributions, data, y)
    
    #利用重要基因进行PCA
    sorted_features_topn_reduced_df = PCA_cluster(data, sorted_features_topn, y)
    
    #利用原始特征进行PCA
    original_features_reduced_df = PCA_cluster_feature(X, y, 1)
    
    
    #利用潜向量进行PCA
    latent_vector_reduced_df = PCA_cluster_feature(get_attended_vectors(model, X_pca), y, 2)






