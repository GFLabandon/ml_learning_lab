# -*- coding: utf-8 -*-
"""
实验四 · 代码一：信息熵与基尼系数函数图像 + 树深度对准确率的影响分析
数据：北京市空气质量数据（2014-2019）
目标：多分类预测空气质量等级（6个等级）
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import warnings

warnings.filterwarnings('ignore')

# ── 0. 中文字体配置（跨平台） ─────────────────────────────────────────────
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = [
    'SimHei', 'STHeiti', 'Noto Sans CJK SC', 'Noto Sans CJK JP',
    'WenQuanYi Zen Hei', 'Arial Unicode MS', 'DejaVu Sans'
]


# ============ 1. 绘制信息熵与基尼系数函数 ============
print("="*60)
print("实验四（代码一）：纯度度量函数 + 树深度影响分析")
print("="*60)

print("\n" + "-"*60)
print("第一部分：绘制纯度度量函数")
print("-"*60)

# 创建图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('信息熵 vs 基尼系数：纯度度量方法对比', fontsize=14, fontweight='bold')

# 概率范围
p = np.linspace(0, 1, 100)

# 信息熵函数
entropy = -np.where(p == 0, 0, p * np.log2(p)) - np.where(1-p == 0, 0, (1-p) * np.log2(1-p))

# 基尼系数函数
gini = 2 * p * (1 - p)

# 绘制信息熵
ax1.plot(p, entropy, 'b-', linewidth=2.5, label='信息熵')
ax1.fill_between(p, entropy, alpha=0.3, color='blue')
ax1.set_xlabel('正例概率 (p)', fontsize=11)
ax1.set_ylabel('信息熵值', fontsize=11)
ax1.set_title('信息熵函数\nH(p) = -p·log₂(p) - (1-p)·log₂(1-p)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1.1])
ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='最大值点')
ax1.legend()

# 绘制基尼系数
ax2.plot(p, gini, 'g-', linewidth=2.5, label='基尼系数')
ax2.fill_between(p, gini, alpha=0.3, color='green')
ax2.set_xlabel('正例概率 (p)', fontsize=11)
ax2.set_ylabel('基尼系数值', fontsize=11)
ax2.set_title('基尼系数函数\nGini(p) = 2p(1-p)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 0.55])
ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='最大值点')
ax2.legend()

plt.tight_layout()
current_dir = os.path.dirname(os.path.abspath(__file__))
output_path1 = os.path.join(current_dir, 'purity_functions.png')
plt.savefig(output_path1, dpi=300, bbox_inches='tight')
print(f"✓ 纯度函数图表已保存：{output_path1}")
plt.close()

# ============ 2. 树深度对准确率的影响分析 ============
print("\n" + "-"*60)
print("第二部分：树深度影响分析")
print("-"*60)

# 加载数据
project_root = os.path.dirname(current_dir)
data_path = os.path.join(project_root, 'data', '北京市空气质量数据.xlsx')
if not os.path.exists(data_path):
    data_path = '北京市空气质量数据.xlsx'

print(f"\n📂 数据路径：{data_path}")

try:
    data = pd.read_excel(data_path)
    print(f"✓ 数据加载成功，样本数：{len(data)}")
except Exception as e:
    print(f"✗ 数据加载失败：{e}")
    exit(1)

# 数据预处理
# 删除质量等级为"无"的记录
data = data[data['质量等级'] != '无'].reset_index(drop=True)
print(f"✓ 清洗后样本数：{len(data)}")

# 特征和目标
features = ['PM2.5', 'PM10', 'SO2', 'CO', 'NO2', 'O3']
X = data[features].values
le = LabelEncoder()
y = le.fit_transform(data['质量等级'])

print(f"✓ 特征数：{len(features)}")
print(f"✓ 目标类别数：{len(le.classes_)}")
print(f"✓ 类别分布：{np.bincount(y)}")

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ 数据分割：训练集{len(X_train)}, 测试集{len(X_test)}")

# 测试不同深度
depths = range(1, 31)
train_accs = []
test_accs = []

print("\n遍历树深度 1-30...")
for depth in depths:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42, criterion='entropy')
    clf.fit(X_train, y_train)
    
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    
    train_accs.append(train_acc)
    test_accs.append(test_acc)

# 找到最优深度
best_depth = depths[np.argmax(test_accs)]
best_test_acc = max(test_accs)

print(f"\n✓ 最优深度：{best_depth}")
print(f"✓ 最优测试准确率：{best_test_acc:.4f}")

# 绘制树深度影响
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(depths, train_accs, 'o-', linewidth=2, markersize=5, label='训练集准确率', color='blue')
ax.plot(depths, test_accs, 's-', linewidth=2, markersize=5, label='测试集准确率', color='red')
ax.axvline(x=best_depth, color='green', linestyle='--', linewidth=2, label=f'最优深度={best_depth}')
ax.scatter([best_depth], [best_test_acc], color='red', s=200, zorder=5, marker='*')

ax.set_xlabel('树深度', fontsize=12)
ax.set_ylabel('准确率', fontsize=12)
ax.set_title('决策树深度对模型准确率的影响', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim([0.5, 1.05])

# 添加注释
ax.annotate(f'最优点\n深度={best_depth}\n准确率={best_test_acc:.4f}',
            xy=(best_depth, best_test_acc),
            xytext=(best_depth+2, best_test_acc-0.05),
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='red'))

plt.tight_layout()
output_path2 = os.path.join(current_dir, 'tree_depth_analysis.png')
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"✓ 树深度分析图表已保存：{output_path2}")
plt.close()

# ============ 3. 对比entropy与gini准则 ============
print("\n" + "-"*60)
print("第三部分：entropy vs gini准则对比")
print("-"*60)

criterions = ['entropy', 'gini']
criterion_names = ['信息熵', 'Gini系数']
colors = ['#2E86AB', '#A23B72']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('不同分割准则对树深度的影响', fontsize=14, fontweight='bold')

for idx, (criterion, criterion_name, color) in enumerate(zip(criterions, criterion_names, colors)):
    train_accs_c = []
    test_accs_c = []
    
    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42, criterion=criterion)
        clf.fit(X_train, y_train)
        
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        
        train_accs_c.append(train_acc)
        test_accs_c.append(test_acc)
    
    best_depth_c = depths[np.argmax(test_accs_c)]
    best_acc_c = max(test_accs_c)
    
    ax = axes[idx]
    ax.plot(depths, train_accs_c, 'o-', linewidth=2, markersize=4, label='训练集', color=color)
    ax.plot(depths, test_accs_c, 's--', linewidth=2, markersize=4, label='测试集', color=color, alpha=0.7)
    ax.axvline(x=best_depth_c, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.scatter([best_depth_c], [best_acc_c], color=color, s=150, zorder=5, marker='*')
    
    ax.set_xlabel('树深度', fontsize=11)
    ax.set_ylabel('准确率', fontsize=11)
    ax.set_title(f'{criterion_name}\n最优深度={best_depth_c}, 准确率={best_acc_c:.4f}', 
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.05])

plt.tight_layout()
output_path3 = os.path.join(current_dir, 'criterion_comparison.png')
plt.savefig(output_path3, dpi=300, bbox_inches='tight')
print(f"✓ 准则对比图表已保存：{output_path3}")
plt.close()

# ============ 总结 ============
print("\n" + "="*60)
print("实验总结")
print("="*60)
print(f"""
✓ 信息熵与基尼系数分析完成

  关键发现：
  • 信息熵和基尼系数都在正例概率为0.5时达到最大值
  • 两种度量方法都能有效评估属性分割的效果
  • Gini系数计算更高效（不涉及对数运算）
  
✓ 树深度影响分析完成

  最优参数：
  • 最优树深度：{best_depth}
  • 测试集最高准确率：{best_test_acc:.4f}
  
  规律发现：
  • 树深度过小：模型欠拟合，训练集和测试集准确率都较低
  • 树深度过大：模型过拟合，训练集准确率很高，测试集准确率反而下降
  • 存在最优的树深度使测试集准确率最大化

✓ 分割准则对比

  • Information Entropy（信息熵）：基于信息论的度量
  • Gini Index（基尼系数）：更高效的计算方法
  • 在实际应用中两者性能相近，选择取决于计算效率考虑
""")
print("="*60)
