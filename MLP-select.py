import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.inspection import permutation_importance


# 数据读取与预处理
def load_data(file_path):
    df = pd.read_excel(file_path, engine='openpyxl')
    df = df.dropna().replace([np.inf, -np.inf], np.nan).dropna()  # 增强异常值处理
    X = df.iloc[:, :71]
    y = df.iloc[:, 71]
    return X, y


# 特征重要性计算（排列重要性）
def calculate_feature_importance(model, X, y, n_repeats=10):
    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=42,
        scoring='neg_mean_squared_error'
    )
    return result.importances_mean


# 主程序
if __name__ == "__main__":
    input_path = r'C:\Users\HW\Desktop\0510番茄\fruity.xlsx'
    output_path = r'C:\Users\HW\Desktop\0510番茄\fruity\fruity_importance.xlsx'

    # 加载数据
    X, y = load_data(input_path)

    # 分类特征处理
    categorical_cols = X.select_dtypes(include=['object']).columns
    numeric_cols = X.select_dtypes(exclude=['object']).columns

    # 构建预处理流程
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 应用预处理
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 超参数优化
    param_grid = {
        'hidden_layer_sizes': [(128, 64), (100, 50), (256,)],
        'alpha': [0.01, 0.1, 0.5],
        'activation': ['tanh', 'relu'],
        'learning_rate_init': [0.001, 0.005]
    }

    mlp = MLPRegressor(
        solver='adam',
        early_stopping=True,
        validation_fraction=0.15,
        max_iter=2000,
        random_state=42
    )

    grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train_processed, y_train)
    best_model = grid_search.best_estimator_

    # 训练过程可视化
    plt.plot(best_model.loss_curve_)
    plt.title("Learning Curve")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

    # 模型评估
    y_pred = best_model.predict(X_test_processed)
    print(f"最佳参数：{grid_search.best_params_}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"R²: {r2_score(y_test, y_pred):.4f}")

    # 特征重要性分析
    importance_scores = calculate_feature_importance(best_model, X_test_processed, y_test)
    feature_names = preprocessor.get_feature_names_out()
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores
    }).sort_values(by='Importance', ascending=False)

    # Excel输出
    importance_df.to_excel(output_path, index=False)
    print(f"特征重要性结果已保存至：{output_path}")