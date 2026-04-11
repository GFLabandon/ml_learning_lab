# -*- coding: utf-8 -*-
"""
多元线性回归模型：基于多个污染物特征预测PM2.5浓度
数据源：北京市空气质量监测数据(2014-2019)
特征：CO、SO2、NO2、O3、PM10
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import os

# ============ 配置中文字体（macOS 优化版） ============
matplotlib.rcParams['axes.unicode_minus'] = False

# macOS 优先使用系统苹方字体，其次才是其他中文字体
plt.rcParams['font.sans-serif'] = [
    'PingFang SC',          # ← macOS 自带，最清晰
    'STHeiti',              # 华文黑体
    'Arial Unicode MS',
    'Noto Sans CJK SC',
    'Noto Sans CJK JP',
    'WenQuanYi Zen Hei',
    'SimHei',
    'DejaVu Sans'
]

# ============ 1. 数据加载与探索 ============
print("=" * 60)
print("实验二：多元线性回归预测PM2.5浓度")
print("=" * 60)

# 确定数据路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
data_path = os.path.join(project_root, 'data', '北京市空气质量数据.xlsx')

if not os.path.exists(data_path):
    data_path = '北京市空气质量数据.xlsx'

print(f"\n📂 数据路径：{data_path}")
print(f"📂 路径是否存在：{os.path.exists(data_path)}")

# 加载数据
try:
    data = pd.read_excel(data_path)
    print(f"\n✓ 数据加载成功")
    print(f"  数据形状：{data.shape}")
    print(f"  列名：{list(data.columns)}")
except Exception as e:
    print(f"✗ 数据加载失败：{e}")
    exit(1)

# ============ 2. 数据预处理 ============
print("\n" + "-" * 60)
print("数据预处理阶段")
print("-" * 60)

# 检测缺失值
features = ['CO', 'SO2', 'NO2', 'O3', 'PM10', 'PM2.5']
null_count = data[features].isnull().sum()
print(f"\n缺失值检查：\n{null_count}")

# 提取特征
X = data[['CO', 'SO2', 'NO2', 'O3', 'PM10']].values
y = data['PM2.5'].values

print(f"\n✓ 提取特征：")
print(f"  自变量形状：{X.shape}")
print(f"  因变量形状：{y.shape}")
print(f"  特征列表：CO, SO2, NO2, O3, PM10")

# 特征统计信息
print(f"\n✓ 特征统计信息：")
feature_names = ['CO', 'SO2', 'NO2', 'O3', 'PM10']
for i, name in enumerate(feature_names):
    print(f"  {name:5s} - 均值:{X[:, i].mean():7.2f}, 标准差:{X[:, i].std():7.2f}")
print(f"  PM2.5 - 均值:{y.mean():7.2f}, 标准差:{y.std():7.2f}")

# ============ 3. 分割训练集和测试集 ============
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n✓ 数据分割：")
print(f"  训练集大小：{X_train.shape[0]} (80%)")
print(f"  测试集大小：{X_test.shape[0]} (20%)")

# ============ 4. 构建和训练模型 ============
print("\n" + "-" * 60)
print("模型构建与训练")
print("-" * 60)

model = LinearRegression()
model.fit(X_train, y_train)

print(f"\n✓ 模型训练完成")
print(f"\n  📊 模型方程：")
print(f"     PM2.5 = ", end="")

# 构建方程字符串
equation_parts = []
for i, name in enumerate(feature_names):
    coef = model.coef_[i]
    sign = "+" if coef >= 0 else ""
    equation_parts.append(f"{sign}{coef:.4f}×{name}")

equation = " ".join(equation_parts) + f" + {model.intercept_:.4f}"
print(equation)

# 打印各特征系数
print(f"\n  各特征系数：")
coef_data = pd.DataFrame({
    '特征': feature_names,
    '系数': model.coef_,
    '绝对值': np.abs(model.coef_)
})
coef_data = coef_data.sort_values('绝对值', ascending=False)
print(coef_data.to_string(index=False))
print(f"  截距：{model.intercept_:.4f}")

# 找出影响最大的特征
max_impact_idx = np.argmax(np.abs(model.coef_))
print(f"\n  ✨ 对PM2.5影响最大的特征：{feature_names[max_impact_idx]}")
print(f"     系数值：{model.coef_[max_impact_idx]:.4f}")

# ============ 5. 模型预测与评估 ============
print("\n" + "-" * 60)
print("模型预测与性能评估")
print("-" * 60)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 计算评估指标
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\n✓ 训练集性能指标：")
print(f"  MSE (均方误差)：{train_mse:.4f}")
print(f"  RMSE (均方根误差)：{train_rmse:.4f}")
print(f"  MAE (平均绝对误差)：{train_mae:.4f}")
print(f"  R² 评分：{train_r2:.4f}")

print(f"\n✓ 测试集性能指标：")
print(f"  MSE (均方误差)：{test_mse:.4f}")
print(f"  RMSE (均方根误差)：{test_rmse:.4f}")
print(f"  MAE (平均绝对误差)：{test_mae:.4f}")
print(f"  R² 评分：{test_r2:.4f}")

# 显示部分预测结果对比
print(f"\n预测结果对比（前10条）：")
comparison_df = pd.DataFrame({
    '真实值': y_test[:10],
    '预测值': y_test_pred[:10],
    '绝对误差': np.abs(y_test[:10] - y_test_pred[:10]),
    '相对误差%': (np.abs(y_test[:10] - y_test_pred[:10]) / y_test[:10] * 100).round(2)
})
print(comparison_df.to_string(index=False))

# ============ 6. 可视化 ============
print("\n" + "-" * 60)
print("生成可视化图表")
print("-" * 60)

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

fig.suptitle('多元线性回归：多污染物浓度预测PM2.5浓度',
             fontsize=16, fontweight='bold', y=0.995)

# 图1-5：各特征与PM2.5的关系
for i, name in enumerate(feature_names):
    ax = fig.add_subplot(gs[i // 3, i % 3])
    ax.scatter(X[:, i], y, alpha=0.4, s=15, color='steelblue')
    # 拟合直线
    z = np.polyfit(X[:, i], y, 1)
    p = np.poly1d(z)
    X_line = np.linspace(X[:, i].min(), X[:, i].max(), 100)
    ax.plot(X_line, p(X_line), "r-", linewidth=2)
    ax.set_xlabel(f'{name}浓度', fontsize=10)
    ax.set_ylabel('PM2.5浓度 (μg/m³)', fontsize=10)
    ax.set_title(f'{name} vs PM2.5 (系数:{model.coef_[i]:.4f})', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

# 图6：特征系数对比
ax6 = fig.add_subplot(gs[1, 2])
colors = ['red' if c < 0 else 'green' for c in model.coef_]
bars = ax6.barh(feature_names, model.coef_, color=colors, alpha=0.7)
ax6.set_xlabel('系数值', fontsize=10)
ax6.set_title('各特征系数对比', fontsize=11, fontweight='bold')
ax6.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax6.grid(True, alpha=0.3, axis='x')
# 添加数值标签
for i, (bar, val) in enumerate(zip(bars, model.coef_)):
    ax6.text(val, i, f' {val:.4f}', va='center', fontsize=9)

# 图7：预测值vs真实值
ax7 = fig.add_subplot(gs[2, 0])
ax7.scatter(y_test, y_test_pred, alpha=0.6, s=30, color='darkorange')
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
ax7.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测')
ax7.set_xlabel('真实PM2.5浓度 (μg/m³)', fontsize=10)
ax7.set_ylabel('预测PM2.5浓度 (μg/m³)', fontsize=10)
ax7.set_title(f'预测值 vs 真实值\n(R²={test_r2:.4f})', fontsize=11, fontweight='bold')
ax7.legend()
ax7.grid(True, alpha=0.3)

# 图8：残差分析
ax8 = fig.add_subplot(gs[2, 1])
residuals = y_test - y_test_pred
ax8.scatter(y_test_pred, residuals, alpha=0.6, s=30, color='purple')
ax8.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax8.set_xlabel('预测PM2.5浓度 (μg/m³)', fontsize=10)
ax8.set_ylabel('残差 (真实-预测)', fontsize=10)
ax8.set_title('残差分析', fontsize=11, fontweight='bold')
ax8.grid(True, alpha=0.3)

# 图9：残差分布直方图
ax9 = fig.add_subplot(gs[2, 2])
ax9.hist(residuals, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
ax9.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax9.set_xlabel('残差值', fontsize=10)
ax9.set_ylabel('频数', fontsize=10)
ax9.set_title(f'残差分布\n(均值:{residuals.mean():.4f}, 标准差:{residuals.std():.4f})',
              fontsize=11, fontweight='bold')
ax9.grid(True, alpha=0.3, axis='y')

# 保存图表
output_path = os.path.join(current_dir, 'multiple_linear_regression_results.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ 图表已保存：{output_path}")

plt.show()

# ============ 7. 总结 ============
print("\n" + "=" * 60)
print("实验总结")
print("=" * 60)

# 相关性分析
print(f"\n✓ 特征与PM2.5的相关性：")
corr_data = []
for i, name in enumerate(feature_names):
    corr = np.corrcoef(X[:, i], y)[0, 1]
    corr_data.append({'特征': name, '相关系数': corr})
corr_df = pd.DataFrame(corr_data).sort_values('相关系数', key=abs, ascending=False)
print(corr_df.to_string(index=False))

print(f"""
✓ 多元线性回归模型建立成功

  模型方程：
  PM2.5 = {equation}

  性能评估（测试集）：
  • R² 分数：{test_r2:.4f}
  • RMSE：{test_rmse:.4f} μg/m³
  • MAE：{test_mae:.4f} μg/m³
  • MSE：{test_mse:.4f}

  主要发现：
  • 最具影响力特征：{feature_names[max_impact_idx]}（系数：{model.coef_[max_impact_idx]:.4f}）
  • 模型R²为{test_r2:.4f}，说明多元模型可解释约{test_r2 * 100:.1f}%的PM2.5变异
  • 相比一元模型，多元模型性能显著提升

  模型可用性：{"优秀" if test_r2 > 0.8 else "很好" if test_r2 > 0.7 else "较好" if test_r2 > 0.6 else "一般"}
""")
print("=" * 60)