# -*- coding: utf-8 -*-
"""
一元线性回归模型：基于CO浓度预测PM2.5浓度
数据源：北京市空气质量监测数据(2014-2019)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
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
print("="*60)
print("实验一：一元线性回归预测PM2.5浓度")
print("="*60)

# 确定数据路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
data_path = os.path.join(project_root, 'data', '北京市空气质量数据.xlsx')

# 如果相对路径不存在，尝试直接使用文件名
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
print("\n" + "-"*60)
print("数据预处理阶段")
print("-"*60)

# 检测缺失值
null_count = data[['CO', 'PM2.5']].isnull().sum()
print(f"\n缺失值检查：\n{null_count}")

# 提取自变量(CO)和因变量(PM2.5)
X = data[['CO']].values
y = data['PM2.5'].values

print(f"\n✓ 提取特征：")
print(f"  自变量(CO)形状：{X.shape}")
print(f"  因变量(PM2.5)形状：{y.shape}")
print(f"  CO 统计信息 - 均值:{X.mean():.2f}, 标准差:{X.std():.2f}")
print(f"  PM2.5 统计信息 - 均值:{y.mean():.2f}, 标准差:{y.std():.2f}")

# ============ 3. 分割训练集和测试集 ============
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n✓ 数据分割：")
print(f"  训练集大小：{X_train.shape[0]} (80%)")
print(f"  测试集大小：{X_test.shape[0]} (20%)")

# ============ 4. 构建和训练模型 ============
print("\n" + "-"*60)
print("模型构建与训练")
print("-"*60)

model = LinearRegression()
model.fit(X_train, y_train)

coef = model.coef_[0]
intercept = model.intercept_

print(f"\n✓ 模型训练完成")
print(f"  模型系数：a = {coef:.4f}")
print(f"  模型截距：b = {intercept:.4f}")
print(f"\n  📊 模型方程：PM2.5 = {coef:.4f} × CO + {intercept:.4f}")

# ============ 5. 模型预测与评估 ============
print("\n" + "-"*60)
print("模型预测与性能评估")
print("-"*60)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 计算评估指标
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\n✓ 训练集性能指标：")
print(f"  MSE (均方误差)：{train_mse:.4f}")
print(f"  RMSE (均方根误差)：{train_rmse:.4f}")
print(f"  R² 评分：{train_r2:.4f}")

print(f"\n✓ 测试集性能指标：")
print(f"  MSE (均方误差)：{test_mse:.4f}")
print(f"  RMSE (均方根误差)：{test_rmse:.4f}")
print(f"  R² 评分：{test_r2:.4f}")

# 显示部分预测结果对比
print(f"\n预测结果对比（前10条）：")
comparison_df = pd.DataFrame({
    '真实值': y_test[:10],
    '预测值': y_test_pred[:10],
    '误差': np.abs(y_test[:10] - y_test_pred[:10])
})
print(comparison_df.to_string(index=False))

# ============ 6. 可视化 ============
print("\n" + "-"*60)
print("生成可视化图表")
print("-"*60)

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('一元线性回归：CO浓度预测PM2.5浓度', fontsize=16, fontweight='bold')

# 图1：训练集散点图与拟合线
ax1 = axes[0, 0]
ax1.scatter(X_train, y_train, alpha=0.5, s=20, color='steelblue', label='训练数据')
# 生成拟合线
X_line = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
y_line = model.predict(X_line)
ax1.plot(X_line, y_line, color='red', linewidth=2, label='拟合直线')
ax1.set_xlabel('CO浓度 (mg/m³)', fontsize=11)
ax1.set_ylabel('PM2.5浓度 (μg/m³)', fontsize=11)
ax1.set_title('训练集：CO vs PM2.5 及拟合直线', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 图2：测试集散点图与拟合线
ax2 = axes[0, 1]
ax2.scatter(X_test, y_test, alpha=0.5, s=20, color='darkorange', label='测试数据')
X_line = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1, 1)
y_line = model.predict(X_line)
ax2.plot(X_line, y_line, color='red', linewidth=2, label='拟合直线')
ax2.set_xlabel('CO浓度 (mg/m³)', fontsize=11)
ax2.set_ylabel('PM2.5浓度 (μg/m³)', fontsize=11)
ax2.set_title('测试集：CO vs PM2.5 及拟合直线', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 图3：预测值vs真实值
ax3 = axes[1, 0]
ax3.scatter(y_test, y_test_pred, alpha=0.6, s=30, color='green', label='预测结果')
# 添加理想拟合线（y=x）
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测')
ax3.set_xlabel('真实PM2.5浓度 (μg/m³)', fontsize=11)
ax3.set_ylabel('预测PM2.5浓度 (μg/m³)', fontsize=11)
ax3.set_title(f'预测值 vs 真实值 (R²={test_r2:.4f})', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 图4：残差分布
ax4 = axes[1, 1]
residuals = y_test - y_test_pred
ax4.scatter(y_test_pred, residuals, alpha=0.6, s=30, color='purple')
ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax4.set_xlabel('预测PM2.5浓度 (μg/m³)', fontsize=11)
ax4.set_ylabel('残差 (真实-预测)', fontsize=11)
ax4.set_title('残差分析', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.tight_layout()

# 保存图表
output_path = os.path.join(current_dir, 'simple_linear_regression_results.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ 图表已保存：{output_path}")

plt.show()

# ============ 7. 总结 ============
print("\n" + "="*60)
print("实验总结")
print("="*60)
print(f"""
✓ 一元线性回归模型建立成功
  
  模型公式：PM2.5 = {coef:.4f} × CO + {intercept:.4f}
  
  性能评估（测试集）：
  • R² 分数：{test_r2:.4f} (范围0-1，越接近1越好)
  • RMSE：{test_rmse:.4f} μg/m³
  • MSE：{test_mse:.4f}
  
  结论：
  {f"• CO浓度与PM2.5呈正相关，系数为{coef:.4f}" if coef > 0 else "• CO浓度与PM2.5呈负相关"}
  • 模型R²为{test_r2:.4f}，说明CO浓度可以解释约{test_r2*100:.1f}%的PM2.5变异
  • 模型可用性：{"较好" if test_r2 > 0.7 else "一般" if test_r2 > 0.5 else "较差"}
""")
print("="*60)
