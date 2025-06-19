import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
import os

# 修改后的文件路径
file_path = r'C:\Users\HW\Desktop\0510番茄\freshness.xlsx'  # 数据文件路径
df = pd.read_excel(file_path)  # 使用 read_excel 读取 Excel 文件

# 提取特征和目标值
X = df.iloc[:, :71].values  # 前71列作为特征
y = df.iloc[:, -1].values.reshape(-1, 1)  # 最后一列作为目标值

# 数据标准化
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# 修改后的输出路径
output_dir = r'C:\Users\HW\Desktop\0510番茄\freshness\R2\MLP'  # 输出路径
os.makedirs(output_dir, exist_ok=True)

# 定义 PyTorch 数据集
class ChemicalDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 定义 MLP 模型
class MLPRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # 输入层到隐藏层
        self.fc2 = nn.Linear(128, 64)  # 隐藏层到隐藏层
        self.fc3 = nn.Linear(64, output_dim)  # 隐藏层到输出层
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 初始化模型和优化器
input_dim = X_scaled.shape[1]  # 特征的数量
output_dim = 1  # 输出维度（目标值）
criterion = nn.MSELoss()

# 顺序向后选择 (SBS)
def sequential_backward_selection(X_train, X_val, y_train, y_val, initial_features):
    num_features = X_train.shape[1]
    selected_features = list(range(num_features))
    results = []

    for step in range(num_features):
        best_mse = float('inf')
        best_r2 = float('-inf')
        best_feature_to_remove = None

        for feature_idx in selected_features:
            temp_features = [f for f in selected_features if f != feature_idx]
            X_train_temp = X_train[:, temp_features]
            X_val_temp = X_val[:, temp_features]

            # 重新计算 input_dim
            input_dim = len(temp_features)
            model = MLPRegressor(input_dim, output_dim)
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            train_dataset = ChemicalDataset(X_train_temp, y_train)
            val_dataset = ChemicalDataset(X_val_temp, y_val)

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

            train_model(model, train_loader, criterion, optimizer, epochs=50)
            mse, r2 = evaluate_model(model, val_loader, y_val)

            if mse < best_mse:
                best_mse = mse
                best_r2 = r2
                best_feature_to_remove = feature_idx

        # 移除效果最好的特征
        selected_features.remove(best_feature_to_remove)
        results.append({
            'Removed Feature': initial_features[best_feature_to_remove],
            'Remaining Features': [initial_features[i] for i in selected_features],
            'MSE': best_mse,
            'R2': best_r2
        })

        # 调试信息
        print(f"Step {step + 1}/{num_features}: Removed Feature: {initial_features[best_feature_to_remove]}, Remaining Features: {len(selected_features)}")

    return results

# k折交叉验证
def k_fold_cross_validation(X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    all_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}/{k}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 获取初始特征名
        initial_features = df.columns[:71].tolist()

        # 执行 SBS
        sbs_results = sequential_backward_selection(X_train, X_val, y_train, y_val, initial_features)
        all_results.append(sbs_results)

    return all_results

# 训练模型
def train_model(model, train_loader, criterion, optimizer, epochs=50):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

# 评估模型
def evaluate_model(model, val_loader, y_val_original):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in val_loader:
            outputs = model(inputs)
            predictions.append(outputs.numpy())

    predictions = np.vstack(predictions)
    predictions = scaler_y.inverse_transform(predictions)  # 反标准化
    y_val_original = scaler_y.inverse_transform(y_val_original)

    mse = mean_squared_error(y_val_original, predictions)
    r2 = r2_score(y_val_original, predictions)

    return mse, r2

# 执行 k 折交叉验证
all_results = k_fold_cross_validation(X_scaled, y_scaled, k=5)

# 将所有结果保存到一个 Excel 文件中
with pd.ExcelWriter(os.path.join(output_dir, 'k_fold_sbs_results.xlsx')) as writer:
    for fold, results in enumerate(all_results):
        sbs_df = pd.DataFrame(results)
        sbs_df.to_excel(writer, sheet_name=f'Fold_{fold + 1}', index=False)

print("所有交叉验证的 SBS 结果已保存到 'k_fold_sbs_results.xlsx' 文件中。")