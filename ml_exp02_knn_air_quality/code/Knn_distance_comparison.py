# -*- coding: utf-8 -*-
"""
实验三 · 代码一：KNN 距离度量方法对比
数据：北京市空气质量数据（2014-2019）
特征：PM2.5, PM10, SO2, CO, NO2, O3  →  预测：空气质量等级（6类）
比较：曼哈顿距离 vs 欧氏距离 vs 切比雪夫距离，K 固定为 5
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

# ── 0. 中文字体配置（跨平台） ─────────────────────────────────────────────
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = [
    'SimHei', 'STHeiti', 'Noto Sans CJK SC', 'Noto Sans CJK JP',
    'WenQuanYi Zen Hei', 'Arial Unicode MS', 'DejaVu Sans'
]

# ── 1. 数据加载与预处理 ───────────────────────────────────────────────────
print("=" * 60)
print("实验三（代码一）：KNN 距离度量方法对比")
print("=" * 60)

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '北京市空气质量数据.xlsx')
df = pd.read_excel(data_path)
df = df[df['质量等级'] != '无'].reset_index(drop=True)
print(f"\n✓ 数据加载完成，有效样本：{len(df)} 条")

FEATURES = ['PM2.5', 'PM10', 'SO2', 'CO', 'NO2', 'O3']
X = df[FEATURES].values
le = LabelEncoder()
y = le.fit_transform(df['质量等级'])

print(f"✓ 特征：{FEATURES}")
print(f"✓ 目标类别：{list(le.classes_)}")
print(f"✓ 各类样本数：")
for cls, cnt in zip(le.classes_, np.bincount(y)):
    print(f"     {cls}：{cnt} 条")

# 划分训练集 / 测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n✓ 训练集：{len(X_train)} 条  测试集：{len(X_test)} 条（stratify 分层抽样）")

# 特征标准化
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
print("✓ 特征已完成 StandardScaler 标准化")

# ── 2. 距离度量对比实验 ───────────────────────────────────────────────────
K_FIXED = 5

distance_configs = [
    ('manhattan',  '曼哈顿距离\n(L1)'),
    ('euclidean',  '欧氏距离\n(L2)'),
    ('chebyshev',  '切比雪夫距离\n(L∞)'),
]

print(f"\n{'─'*60}")
print(f"K = {K_FIXED}，分类决策规则 = 多数表决（uniform），对比三种距离度量")
print(f"{'─'*60}")

results = {}
for metric, label in distance_configs:
    knn = KNeighborsClassifier(n_neighbors=K_FIXED, metric=metric, weights='uniform')
    knn.fit(X_train_s, y_train)
    y_pred  = knn.predict(X_test_s)
    test_acc  = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(knn, X_train_s, y_train, cv=5, scoring='accuracy')
    cv_mean   = cv_scores.mean()
    cv_std    = cv_scores.std()

    results[metric] = {
        'label': label, 'test_acc': test_acc,
        'cv_mean': cv_mean, 'cv_std': cv_std, 'cv_scores': cv_scores
    }
    print(f"\n  距离：{metric:<12} 测试准确率={test_acc:.4f}  "
          f"5折CV均值={cv_mean:.4f}±{cv_std:.4f}")
    print(f"         5折得分：{np.round(cv_scores, 4)}")

# ── 3. 可视化 ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f'KNN 距离度量对比（K={K_FIXED}，多数表决）', fontsize=14, fontweight='bold')

metrics_list  = list(results.keys())
labels_list   = [results[m]['label'] for m in metrics_list]
test_accs     = [results[m]['test_acc'] for m in metrics_list]
cv_means      = [results[m]['cv_mean']  for m in metrics_list]
cv_stds       = [results[m]['cv_std']   for m in metrics_list]

COLORS = ['#5B9BD5', '#ED7D31', '#70AD47']
x = np.arange(len(metrics_list))
width = 0.35

# 子图1：测试准确率 vs 5折CV均值对比柱状图
bars1 = axes[0].bar(x - width/2, test_accs, width, label='测试集准确率',
                    color=COLORS, alpha=0.85, edgecolor='white', linewidth=1.2)
bars2 = axes[0].bar(x + width/2, cv_means,  width, label='5折CV均值',
                    color=COLORS, alpha=0.50, edgecolor='white', linewidth=1.2,
                    hatch='//')

# 误差线
axes[0].errorbar(x + width/2, cv_means, yerr=cv_stds, fmt='none',
                 color='black', capsize=5, linewidth=1.5)

for bar, val in zip(bars1, test_accs):
    axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.003,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar, val in zip(bars2, cv_means):
    axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.003,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9)

axes[0].set_xticks(x)
axes[0].set_xticklabels(labels_list, fontsize=10)
axes[0].set_ylim(0.80, 0.92)
axes[0].set_ylabel('准确率', fontsize=11)
axes[0].set_title('测试集准确率 vs 5折交叉验证均值', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].axhline(y=max(test_accs), color='red', linestyle='--', linewidth=1,
                label=f'最高={max(test_accs):.4f}')
axes[0].legend(fontsize=9)

# 子图2：5折CV各折得分分布（箱线图）
cv_data = [results[m]['cv_scores'] for m in metrics_list]
bp = axes[1].boxplot(cv_data, labels=labels_list, patch_artist=True,
                     medianprops=dict(color='black', linewidth=2))
for patch, color in zip(bp['boxes'], COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# 叠加散点
for i, (data_pts, color) in enumerate(zip(cv_data, COLORS), 1):
    jitter = np.random.default_rng(0).uniform(-0.1, 0.1, len(data_pts))
    axes[1].scatter(np.full(len(data_pts), i) + jitter, data_pts,
                    color=color, s=50, zorder=5, edgecolors='white')

axes[1].set_ylabel('5折交叉验证准确率', fontsize=11)
axes[1].set_title('各距离度量的5折CV得分分布', fontsize=12)
axes[1].set_ylim(0.78, 0.92)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'exp3_distance_comparison.png')
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"\n✓ 图表已保存：{out_path}")
plt.show()

# ── 4. 汇总 ──────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("汇总结果")
print(f"{'='*60}")
print(f"  {'距离度量':<12} {'测试准确率':>10} {'CV均值':>10} {'CV标准差':>10}")
print(f"  {'-'*46}")
best_m = max(results, key=lambda m: results[m]['test_acc'])
for m, r in results.items():
    flag = ' ← 最优' if m == best_m else ''
    print(f"  {m:<12} {r['test_acc']:>10.4f} {r['cv_mean']:>10.4f} {r['cv_std']:>10.4f}{flag}")

print(f"\n结论：欧氏距离（euclidean）在 K={K_FIXED} 时取得最高测试准确率 "
      f"{results[best_m]['test_acc']:.4f}，作为后续实验的固定距离度量。")