# -*- coding: utf-8 -*-
"""
实验一：基于一元线性回归的 PM2.5 浓度预测
数据：北京市空气质量数据（2014-2019年）
自变量：CO（一氧化碳浓度）
因变量：PM2.5
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ── 0. 全局绘图设置（防止中文乱码）────────────────────────────────────────
matplotlib.rcParams['axes.unicode_minus'] = False
# 优先使用 SimHei（Windows/macOS 中文黑体），若不存在则退回系统默认
plt.rcParams['font.sans-serif'] = [
    'Noto Sans CJK SC', 'Noto Sans CJK JP', 'WenQuanYi Zen Hei',
    'SimHei', 'STHeiti', 'Arial Unicode MS', 'DejaVu Sans'
]

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))  # 图片保存到脚本同目录

# ── 1. 导入数据 ───────────────────────────────────────────────────────────
#ata_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
#                         '北京市空气质量数据.xlsx')
# 项目根目录（ml_exp01_air_quality）
# 项目根目录（ml_exp01_air_quality）
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_path = os.path.join(project_root, 'data', '北京市空气质量数据.xlsx')
data = pd.read_excel(data_path)
print("数据集形状：", data.shape)
print(data.head(5))

# ── 2. 数据预处理 ─────────────────────────────────────────────────────────
# 2.1 检测缺失值
print("\n缺失值统计：")
print(data.isnull().sum())

# 2.2 剔除异常/缺失行（质量等级为"无"视为缺失）
data = data[data['质量等级'] != '无'].reset_index(drop=True)
print(f"\n清洗后样本数：{len(data)}")

# ── 3. 探索性数据分析：CO 与 PM2.5 的散点图 ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('实验一：CO 与 PM2.5 关系探索', fontsize=14, fontweight='bold')

axes[0].scatter(data['CO'], data['PM2.5'], alpha=0.3, s=10, color='steelblue')
axes[0].set_xlabel('CO（mg/m³）')
axes[0].set_ylabel('PM2.5（μg/m³）')
axes[0].set_title('原始数据散点图（CO vs PM2.5）')
axes[0].grid(True, linestyle='--', alpha=0.5)

# 相关系数
corr = np.corrcoef(data['CO'], data['PM2.5'])[0, 1]
axes[0].text(0.05, 0.93, f'Pearson r = {corr:.4f}',
             transform=axes[0].transAxes,
             fontsize=10, color='red',
             bbox=dict(facecolor='white', edgecolor='red', alpha=0.7))

# 2.3 筛选线性相关性较强的子集（参考实验文档做法）
data_filtered = data[data['CO'] <= 6].reset_index(drop=True)
axes[1].scatter(data_filtered['CO'], data_filtered['PM2.5'],
                alpha=0.3, s=10, color='darkorange')
axes[1].set_xlabel('CO（mg/m³）')
axes[1].set_ylabel('PM2.5（μg/m³）')
axes[1].set_title('筛选后散点图（CO ≤ 6 mg/m³）')
axes[1].grid(True, linestyle='--', alpha=0.5)
corr2 = np.corrcoef(data_filtered['CO'], data_filtered['PM2.5'])[0, 1]
axes[1].text(0.05, 0.93, f'Pearson r = {corr2:.4f}',
             transform=axes[1].transAxes,
             fontsize=10, color='red',
             bbox=dict(facecolor='white', edgecolor='red', alpha=0.7))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'exp1_scatter.png'), dpi=150)
print("\n[图1] 散点图已保存：exp1_scatter.png")
plt.show()

# ── 4. 生成自变量与因变量 ─────────────────────────────────────────────────
X = data_filtered[['CO']].values
y = data_filtered['PM2.5'].values
print(f"\n筛选后样本数：{len(X)}")
print(f"CO 与 PM2.5 相关系数：{corr2:.4f}")

# ── 5. 拆分训练集与测试集 ─────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"\n训练集：{X_train.shape}，测试集：{X_test.shape}")

# ── 6. 构建三种参数的简单线性回归模型 ────────────────────────────────────
models = {
    '模型1\nnormalize=True, fit_intercept=True':
        LinearRegression(fit_intercept=True),
    '模型2\nnormalize=False, fit_intercept=True':
        LinearRegression(fit_intercept=True),
    '模型3\nnormalize=False, fit_intercept=False':
        LinearRegression(fit_intercept=False),
}

results = {}
for name, reg in models.items():
    # 模型1 手动标准化（sklearn ≥1.0 已移除 normalize 参数，用 StandardScaler 替代）
    if '模型1' in name:
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train)
        Xte = scaler.transform(X_test)
    else:
        Xtr, Xte = X_train, X_test

    reg.fit(Xtr, y_train)
    y_pred = reg.predict(Xte)
    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    coef = reg.coef_[0]
    intercept = reg.intercept_ if reg.fit_intercept else 0.0

    if reg.fit_intercept:
        expr = f'PM2.5 = {coef:.2f} × CO + ({intercept:.2f})'
    else:
        expr = f'PM2.5 = {coef:.2f} × CO'

    results[name] = dict(model=reg, scaler=scaler if '模型1' in name else None,
                         coef=coef, intercept=intercept,
                         mse=mse, r2=r2, expr=expr,
                         Xte=Xte, y_pred=y_pred)
    print(f"\n{'='*50}")
    print(f"【{name.replace(chr(10), ' ')}】")
    print(f"  决策方程：{expr}")
    print(f"  MSE  = {mse:.2f}")
    print(f"  R²   = {r2:.4f}")

# ── 7. 可视化：拟合曲线对比 ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('实验一：三种模型拟合结果对比（测试集）', fontsize=13, fontweight='bold')

colors = ['blue', 'green', 'purple']
for ax, (name, res), color in zip(axes, results.items(), colors):
    # 散点（真实值）
    ax.scatter(X_test, y_test, color='red', alpha=0.4, s=15, label='真实值')
    # 拟合线
    x_line = np.linspace(X_test.min(), X_test.max(), 200).reshape(-1, 1)
    if res['scaler']:
        x_line_scaled = res['scaler'].transform(x_line)
        y_line = res['model'].predict(x_line_scaled)
    else:
        y_line = res['model'].predict(x_line)
    ax.plot(x_line, y_line, color=color, linewidth=2, label='拟合线')

    ax.set_xlabel('CO（mg/m³）')
    ax.set_ylabel('PM2.5（μg/m³）')
    short_name = name.split('\n')[0]
    ax.set_title(f'{short_name}\nMSE={res["mse"]:.1f}  R²={res["r2"]:.4f}')
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.text(0.03, 0.97, res['expr'], transform=ax.transAxes,
            fontsize=8, verticalalignment='top',
            bbox=dict(facecolor='lightyellow', edgecolor='gray', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'exp1_fit_curves.png'), dpi=150)
print("\n[图2] 拟合曲线图已保存：exp1_fit_curves.png")
plt.show()

# ── 8. 残差分析（最优模型：模型2）────────────────────────────────────────
best_name = '模型2\nnormalize=False, fit_intercept=True'
best_res  = results[best_name]

y_pred_best = best_res['y_pred']
residuals   = y_test - y_pred_best

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle('实验一：最优模型残差分析（模型2）', fontsize=13, fontweight='bold')

# 残差分布直方图
axes[0].hist(residuals, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].axvline(0, color='red', linestyle='--')
axes[0].set_xlabel('残差')
axes[0].set_ylabel('频数')
axes[0].set_title('残差分布直方图')

# 残差 vs 预测值
axes[1].scatter(y_pred_best, residuals, alpha=0.4, s=15, color='coral')
axes[1].axhline(0, color='red', linestyle='--')
axes[1].set_xlabel('预测值（PM2.5）')
axes[1].set_ylabel('残差')
axes[1].set_title('残差 vs 预测值')
axes[1].grid(True, linestyle='--', alpha=0.4)

# 真实值 vs 预测值对比
axes[2].scatter(y_test, y_pred_best, alpha=0.4, s=15, color='mediumpurple')
lim = [min(y_test.min(), y_pred_best.min()) - 5,
       max(y_test.max(), y_pred_best.max()) + 5]
axes[2].plot(lim, lim, 'r--', linewidth=1.5, label='理想拟合线（y=x）')
axes[2].set_xlabel('真实 PM2.5（μg/m³）')
axes[2].set_ylabel('预测 PM2.5（μg/m³）')
axes[2].set_title('真实值 vs 预测值')
axes[2].legend(fontsize=9)
axes[2].grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'exp1_residuals.png'), dpi=150)
print("[图3] 残差分析图已保存：exp1_residuals.png")
plt.show()

# ── 9. 汇总输出 ───────────────────────────────────────────────────────────
print("\n" + "="*60)
print("【实验一总结】一元线性回归：CO → PM2.5")
print("="*60)
print(f"{'模型':<10} {'决策方程':<35} {'MSE':>12} {'R²':>8}")
print("-"*70)
for name, res in results.items():
    short = name.split('\n')[0]
    print(f"{short:<10} {res['expr']:<35} {res['mse']:>12.2f} {res['r2']:>8.4f}")
print("\n结论：模型2（fit_intercept=True, 不标准化）与模型1性能相同；")
print("      模型3（无截距）R²最低，MSE最大，拟合效果最差；")
print("      最优模型方程：", results[best_name]['expr'])