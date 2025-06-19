import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
import os

# 1. 数据加载和预处理
file_path = r'C:\Users\HW\Desktop\0510番茄\freshness.xlsx'  # 原始文件路径
df = pd.read_excel(file_path)  # 使用 read_excel 读取 Excel 文件

# 提取特征和目标值
X = df.iloc[:, :71].values  # 前71列作为特征
y = df.iloc[:, -1].values  # 最后一列作为目标值

# 数据标准化
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# 2. 初始化SVR模型
svr = SVR()

# 3. 超参数优化
param_grid = {
    'C': [0.1, 1, 10],
    'epsilon': [0.01, 0.1, 0.2],
    'kernel': ['linear', 'rbf']
}

grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# 4. 评估模型
best_svr = grid_search.best_estimator_
y_pred = best_svr.predict(X_test)
initial_r2 = r2_score(y_test, y_pred)
print(f"初始模型的R²: {initial_r2:.4f}")

# 5. 逐步移除特征并重新训练模型
results = []
feature_names = df.columns[:71].tolist()
coefficients = best_svr.coef_[0] if best_svr.kernel == 'linear' else np.zeros(X_train.shape[1])

while len(feature_names) > 1:
    # 找到系数绝对值最小的特征
    least_important_feature = feature_names[np.argmin(np.abs(coefficients))]
    print(f"移除特征: {least_important_feature}")

    # 移除该特征
    feature_names.remove(least_important_feature)
    coefficients = np.delete(coefficients, np.argmin(np.abs(coefficients)))

    # 重新训练模型
    X_train_reduced = X_train[:, [feature_names.index(name) for name in feature_names]]
    X_test_reduced = X_test[:, [feature_names.index(name) for name in feature_names]]

    best_svr.fit(X_train_reduced, y_train)
    y_pred_reduced = best_svr.predict(X_test_reduced)
    r2_reduced = r2_score(y_test, y_pred_reduced)
    print(f"移除特征后的R²: {r2_reduced:.4f}")

    # 记录结果
    results.append({
        'Removed Feature': least_important_feature,
        'Remaining Features': len(feature_names),
        'R²': r2_reduced
    })

# 6. 输出结果到 CSV 文件
output_dir = r'C:\Users\HW\Desktop\0510番茄'  # 原始输出路径
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, 'svr_feature_reduction_results.csv')
try:
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"特征逐步移除的结果已保存到 '{output_file}' 文件中。")
except PermissionError as e:
    print(f"权限错误，无法写入文件：{e}")
    print("请确保文件未被其他程序占用，并且程序有权限写入目标目录。")