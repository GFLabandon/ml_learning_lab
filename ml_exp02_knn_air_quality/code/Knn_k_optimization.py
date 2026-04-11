# -*- coding: utf-8 -*-
"""
实验三 · 代码二：KNN K 值寻优
数据：北京市空气质量数据（2014-2019）
固定距离度量 = 欧氏距离，遍历 K = 1…50，以5折交叉验证找到最优 K
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
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
print("实验三（代码二）：KNN K 值寻优")
print("=" * 60)

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '北京市空气质量数据.xlsx')
df = pd.read_excel(data_path)
df = df[df['质量等级'] != '无'].reset_index(drop=True)
print(f"\n✓ 有效样本：{len(df)} 条")

FEATURES = ['PM2.5', 'PM10', 'SO2', 'CO', 'NO2', 'O3']
X = df[FEATURES].values
le = LabelEncoder()
y = le.fit_transform(df['质量等级'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
print(f"✓ 训练集：{len(X_train)}  测试集：{len(X_test)}")

# ── 2. K 值遍历（K = 1 ~ 50） ────────────────────────────────────────────
K_RANGE   = range(1, 51)
METRIC    = 'euclidean'
CV_FOLDS  = 5

cv_means, cv_stds, test_accs = [], [], []

print(f"\n正在遍历 K = 1 ~ {max(K_RANGE)}（固定欧氏距离，{CV_FOLDS}折交叉验证）…")
for k in K_RANGE:
    knn = KNeighborsClassifier(n_neighbors=k, metric=METRIC, weights='uniform')
    cv  = cross_val_score(knn, X_train_s, y_train, cv=CV_FOLDS, scoring='accuracy')
    knn.fit(X_train_s, y_train)
    ta  = accuracy_score(y_test, knn.predict(X_test_s))
    cv_means.append(cv.mean())
    cv_stds.append(cv.std())
    test_accs.append(ta)

cv_means  = np.array(cv_means)
cv_stds   = np.array(cv_stds)
test_accs = np.array(test_accs)

best_k_idx = int(np.argmax(cv_means))
best_k     = list(K_RANGE)[best_k_idx]
best_cv    = cv_means[best_k_idx]
best_test  = test_accs[best_k_idx]

print(f"✓ 遍历完成")
print(f"\n最优 K = {best_k}")
print(f"  5折CV准确率：{best_cv:.4f} ± {cv_stds[best_k_idx]:.4f}")
print(f"  对应测试集准确率：{best_test:.4f}")

# 输出 Top-5 K 值
top5_idx = np.argsort(cv_means)[::-1][:5]
print(f"\nTop-5 K 值（按5折CV降序）：")
print(f"  {'K值':>5} {'5折CV均值':>12} {'5折CV标准差':>12} {'测试准确率':>12}")
print(f"  {'-'*45}")
for i in top5_idx:
    k_val = list(K_RANGE)[i]
    marker = ' ★' if k_val == best_k else ''
    print(f"  {k_val:>5} {cv_means[i]:>12.4f} {cv_stds[i]:>12.4f} {test_accs[i]:>12.4f}{marker}")

# ── 3. 可视化 ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'KNN K 值寻优（欧氏距离，{CV_FOLDS}折交叉验证）', fontsize=14, fontweight='bold')

ks = list(K_RANGE)

# 子图1：CV准确率折线 + 置信带
ax = axes[0]
ax.plot(ks, cv_means, 'o-', color='#2E75B6', linewidth=2, markersize=4, label='5折CV均值')
ax.fill_between(ks,
                cv_means - cv_stds,
                cv_means + cv_stds,
                alpha=0.18, color='#2E75B6', label='±1 标准差')
ax.plot(ks, test_accs, 's--', color='#ED7D31', linewidth=1.5, markersize=3,
        alpha=0.75, label='测试集准确率')
ax.axvline(x=best_k, color='red', linestyle='--', linewidth=1.5, label=f'最优K={best_k}')
ax.scatter([best_k], [best_cv], color='red', zorder=10, s=100,
           marker='*', label=f'最优CV={best_cv:.4f}')

ax.set_xlabel('K 值', fontsize=12)
ax.set_ylabel('准确率', fontsize=12)
ax.set_title('K 值 vs 5折交叉验证准确率', fontsize=12)
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max(ks)+1)
ax.set_ylim(max(0.72, cv_means.min()-0.03), min(1.0, cv_means.max()+0.03))

# 子图2：前20个K的柱状图（细节放大）
ax2 = axes[1]
ks20 = ks[:20]
bar_colors = ['#C00000' if k == best_k else '#5B9BD5' for k in ks20]
bars = ax2.bar(ks20, cv_means[:20], color=bar_colors, alpha=0.8,
               edgecolor='white', linewidth=0.8)
ax2.plot(ks20, test_accs[:20], 's--', color='#ED7D31', linewidth=1.5,
         markersize=5, label='测试集准确率', zorder=5)

for bar, val in zip(bars, cv_means[:20]):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.002,
             f'{val:.3f}', ha='center', va='bottom', fontsize=7)

ax2.axvline(x=best_k, color='red', linestyle='--', linewidth=1.5)
ax2.set_xlabel('K 值（K = 1 ~ 20 细节）', fontsize=12)
ax2.set_ylabel('准确率', fontsize=12)
ax2.set_title('K = 1~20 交叉验证准确率（柱状图）', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(max(0.75, cv_means[:20].min()-0.03), min(1.0, cv_means[:20].max()+0.04))

# 标注最优K
ax2.annotate(f'最优 K={best_k}\nCV={best_cv:.4f}',
             xy=(best_k, cv_means[best_k_idx]),
             xytext=(best_k + 2, cv_means[best_k_idx] - 0.015),
             fontsize=9, color='red', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'exp3_k_optimization.png')
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"\n✓ 图表已保存：{out_path}")
plt.show()

# ── 4. 汇总 ──────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("实验结论")
print(f"{'='*60}")
print(f"""
  • 采用5折交叉验证在训练集上对 K = 1~50 逐一评估。
  • 最优参数：K = {best_k}，5折CV准确率 = {best_cv:.4f}。
  • K 过小（K=1~2）时，模型噪声敏感、方差大；
    K 过大（K > 20）时，模型偏差增大，准确率下降。
  • K = {best_k} 为偏差-方差权衡的最优折中点。
  • 后续实验将固定 K = {best_k}，进一步对比分类决策规则。
""")