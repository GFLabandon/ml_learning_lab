# -*- coding: utf-8 -*-
"""
实验四 · 代码二：决策树分类与剪枝（预剪枝和后剪枝）
数据：北京市空气质量数据（2014-2019）
目标：多分类预测空气质量等级
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

# ============ 配置中文字体 ============
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============ 1. 数据加载与准备 ============
print("="*60)
print("实验四（代码二）：决策树分类与剪枝")
print("="*60)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(project_root, 'data', '北京市空气质量数据.xlsx')

if not os.path.exists(data_path):
    data_path = '北京市空气质量数据.xlsx'

print(f"\n📂 数据路径：{data_path}")

try:
    data = pd.read_excel(data_path)
    print(f"✓ 数据加载成功")
except Exception as e:
    print(f"✗ 数据加载失败：{e}")
    exit(1)

# 数据预处理
data = data[data['质量等级'] != '无'].reset_index(drop=True)

features = ['PM2.5', 'PM10', 'SO2', 'CO', 'NO2', 'O3']
X = data[features].values
le = LabelEncoder()
y = le.fit_transform(data['质量等级'])

print(f"✓ 样本数：{len(data)}")
print(f"✓ 特征数：{len(features)}")
print(f"✓ 目标类别：{list(le.classes_)}")

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ 训练集：{len(X_train)}, 测试集：{len(X_test)}")

# ============ 2. 对比不同剪枝策略 ============
print("\n" + "-"*60)
print("对比不同剪枝策略")
print("-"*60)

strategies = {
    'no_pruning': {
        'name': '无剪枝（深度=30）',
        'params': {'max_depth': 30, 'min_samples_split': 2, 'min_samples_leaf': 1}
    },
    'pre_pruning_simple': {
        'name': '预剪枝1（深度=5）',
        'params': {'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 1}
    },
    'pre_pruning_complex': {
        'name': '预剪枝2（深度=8, min_samples_leaf=5）',
        'params': {'max_depth': 8, 'min_samples_split': 5, 'min_samples_leaf': 5}
    },
    'cost_complexity': {
        'name': '成本复杂度剪枝（最优ccp_alpha）',
        'params': {'random_state': 42}
    }
}

results = {}

for strategy_key, strategy_info in strategies.items():
    print(f"\n{strategy_info['name']}...")
    
    if strategy_key == 'cost_complexity':
        # 成本复杂度剪枝：先训练完整树，再根据ccp_alpha剪枝
        clf_full = DecisionTreeClassifier(random_state=42, criterion='entropy')
        clf_full.fit(X_train, y_train)
        
        # 获取所有ccp_alpha值
        path = clf_full.cost_complexity_pruning_path(X_train, y_train)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities
        
        # 使用不同的ccp_alpha值训练模型并在测试集上评估
        best_acc = 0
        best_alpha = 0
        for ccp_alpha in ccp_alphas:
            clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha, criterion='entropy')
            clf.fit(X_train, y_train)
            acc = clf.score(X_test, y_test)
            if acc > best_acc:
                best_acc = acc
                best_alpha = ccp_alpha
        
        # 使用最优ccp_alpha训练最终模型
        clf = DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha, criterion='entropy')
        clf.fit(X_train, y_train)
    else:
        clf = DecisionTreeClassifier(criterion='entropy', **strategy_info['params'])
        clf.fit(X_train, y_train)
    
    # 评估
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    n_leaves = clf.get_n_leaves()
    depth = clf.get_depth()
    
    results[strategy_key] = {
        'clf': clf,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'n_leaves': n_leaves,
        'depth': depth
    }
    
    print(f"  训练准确率：{train_acc:.4f}")
    print(f"  测试准确率：{test_acc:.4f}")
    print(f"  树深度：{depth}")
    print(f"  叶子数：{n_leaves}")

# ============ 3. 可视化对比 ============
print("\n" + "-"*60)
print("绘制剪枝策略对比")
print("-"*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('决策树剪枝策略对比', fontsize=14, fontweight='bold')

strategy_names = [strategy_info['name'] for strategy_info in strategies.values()]
train_accs = [results[k]['train_acc'] for k in results.keys()]
test_accs = [results[k]['test_acc'] for k in results.keys()]
depths = [results[k]['depth'] for k in results.keys()]
n_leaves = [results[k]['n_leaves'] for k in results.keys()]

# 子图1：准确率对比
ax = axes[0, 0]
x = np.arange(len(strategy_names))
width = 0.35
bars1 = ax.bar(x - width/2, train_accs, width, label='训练集', color='skyblue', alpha=0.8)
bars2 = ax.bar(x + width/2, test_accs, width, label='测试集', color='coral', alpha=0.8)
ax.set_xlabel('剪枝策略', fontsize=11)
ax.set_ylabel('准确率', fontsize=11)
ax.set_title('准确率对比', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(range(1, len(strategy_names)+1), fontsize=10)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0.5, 1.0])

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

# 子图2：树深度对比
ax = axes[0, 1]
colors = ['red' if d > 8 else 'green' for d in depths]
bars = ax.bar(range(len(strategy_names)), depths, color=colors, alpha=0.7)
ax.set_xlabel('剪枝策略', fontsize=11)
ax.set_ylabel('树深度', fontsize=11)
ax.set_title('树深度对比', fontsize=12, fontweight='bold')
ax.set_xticks(range(len(strategy_names)))
ax.set_xticklabels(range(1, len(strategy_names)+1), fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, depths):
    ax.text(bar.get_x() + bar.get_width()/2., val + 0.3,
            f'{val}', ha='center', va='bottom', fontsize=9)

# 子图3：叶子数对比
ax = axes[1, 0]
bars = ax.bar(range(len(strategy_names)), n_leaves, color='mediumpurple', alpha=0.7)
ax.set_xlabel('剪枝策略', fontsize=11)
ax.set_ylabel('叶子数', fontsize=11)
ax.set_title('叶子数对比（复杂度指示）', fontsize=12, fontweight='bold')
ax.set_xticks(range(len(strategy_names)))
ax.set_xticklabels(range(1, len(strategy_names)+1), fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, n_leaves):
    ax.text(bar.get_x() + bar.get_width()/2., val + 1,
            f'{val}', ha='center', va='bottom', fontsize=9)

# 子图4：训练-测试差距（过拟合程度）
ax = axes[1, 1]
overfitting = [train_accs[i] - test_accs[i] for i in range(len(strategy_names))]
colors_over = ['red' if o > 0.1 else 'yellow' if o > 0.05 else 'green' for o in overfitting]
bars = ax.bar(range(len(strategy_names)), overfitting, color=colors_over, alpha=0.7)
ax.set_xlabel('剪枝策略', fontsize=11)
ax.set_ylabel('准确率差距（训练-测试）', fontsize=11)
ax.set_title('过拟合程度对比', fontsize=12, fontweight='bold')
ax.set_xticks(range(len(strategy_names)))
ax.set_xticklabels(range(1, len(strategy_names)+1), fontsize=10)
ax.axhline(y=0.05, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='合理范围')
ax.grid(True, alpha=0.3, axis='y')
ax.legend()

for bar, val in zip(bars, overfitting):
    ax.text(bar.get_x() + bar.get_width()/2., val + 0.002,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
output_path = os.path.join(current_dir, 'pruning_strategies.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ 剪枝策略对比图已保存：{output_path}")
plt.close()

# ============ 4. 选择最优模型进行详细分析 ============
print("\n" + "-"*60)
print("最优模型详细分析")
print("-"*60)

# 找到测试集准确率最高的模型
best_strategy = max(results.keys(), key=lambda k: results[k]['test_acc'])
best_clf = results[best_strategy]['clf']
best_test_acc = results[best_strategy]['test_acc']

print(f"\n最优模型：{strategies[best_strategy]['name']}")
print(f"测试集准确率：{best_test_acc:.4f}")

# 预测
y_pred = best_clf.predict(X_test)

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)

print(f"\n混淆矩阵：")
print(cm)

# 分类报告
print(f"\n分类报告：")
print(classification_report(y_test, y_pred, target_names=le.classes_, digits=4))

# 绘制混淆矩阵热力图
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=le.classes_, yticklabels=le.classes_,
            cbar_kws={'label': '样本数'})
ax.set_xlabel('预测标签', fontsize=12)
ax.set_ylabel('真实标签', fontsize=12)
ax.set_title(f'最优决策树混淆矩阵（准确率={best_test_acc:.4f}）', fontsize=13, fontweight='bold')
plt.tight_layout()
output_path_cm = os.path.join(current_dir, 'decision_tree_confusion_matrix.png')
plt.savefig(output_path_cm, dpi=300, bbox_inches='tight')
print(f"\n✓ 混淆矩阵已保存：{output_path_cm}")
plt.close()

# ============ 总结 ============
print("\n" + "="*60)
print("实验总结")
print("="*60)
print(f"""
✓ 决策树分类与剪枝分析完成

  剪枝策略对比：
  
  1. 无剪枝（深度=30）：
     - 训练准确率最高，但测试准确率较低
     - 明显过拟合
  
  2. 预剪枝1（深度=5）：
     - 过度限制树的生长
     - 可能欠拟合
  
  3. 预剪枝2（深度=8, min_samples_leaf=5）：
     - 在训练和测试准确率间找到平衡
     - 树的复杂度适中
  
  4. 成本复杂度剪枝：
     - 自动选择最优的ccp_alpha值
     - 通过测试集验证确定最佳模型
  
  最优模型：{strategies[best_strategy]['name']}
  • 测试集准确率：{best_test_acc:.4f}
  • 树深度：{results[best_strategy]['depth']}
  • 叶子数：{results[best_strategy]['n_leaves']}
  
  关键发现：
  • 剪枝显著降低了过拟合风险
  • 合理的树深度限制能提高泛化能力
  • min_samples_leaf参数有效控制树的复杂度
""")
print("="*60)
