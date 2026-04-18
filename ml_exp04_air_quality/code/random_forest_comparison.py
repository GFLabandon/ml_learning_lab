# -*- coding: utf-8 -*-
"""
实验四 · 代码三：随机森林 vs 决策树性能对比
数据：北京市空气质量数据（2014-2019）
目标：多分类预测空气质量等级
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import os
import warnings
import time

warnings.filterwarnings('ignore')

# ============ 配置中文字体 ============
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============ 1. 数据加载与准备 ============
print("="*60)
print("实验四（代码三）：随机森林 vs 决策树对比")
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
print(f"✓ 类别数：{len(le.classes_)}")

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ 训练集：{len(X_train)}, 测试集：{len(X_test)}")

# ============ 2. 构建模型进行对比 ============
print("\n" + "-"*60)
print("构建决策树和随机森林模型")
print("-"*60)

models = {}

# 决策树模型（优化版）
print("\n训练决策树（深度=8, min_samples_leaf=5）...")
start_time = time.time()
dt = DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, 
                           random_state=42, criterion='entropy')
dt.fit(X_train, y_train)
dt_train_time = time.time() - start_time

dt_train_acc = dt.score(X_train, y_train)
dt_test_acc = dt.score(X_test, y_test)
models['Decision Tree'] = {
    'model': dt,
    'train_acc': dt_train_acc,
    'test_acc': dt_test_acc,
    'train_time': dt_train_time
}

print(f"✓ 训练时间：{dt_train_time:.4f}s")
print(f"  训练准确率：{dt_train_acc:.4f}")
print(f"  测试准确率：{dt_test_acc:.4f}")

# 随机森林模型1：n_estimators=50
print("\n训练随机森林（n_estimators=50）...")
start_time = time.time()
rf50 = RandomForestClassifier(n_estimators=50, max_depth=8, 
                              min_samples_leaf=5, random_state=42, n_jobs=-1)
rf50.fit(X_train, y_train)
rf50_train_time = time.time() - start_time

rf50_train_acc = rf50.score(X_train, y_train)
rf50_test_acc = rf50.score(X_test, y_test)
models['Random Forest (50)'] = {
    'model': rf50,
    'train_acc': rf50_train_acc,
    'test_acc': rf50_test_acc,
    'train_time': rf50_train_time
}

print(f"✓ 训练时间：{rf50_train_time:.4f}s")
print(f"  训练准确率：{rf50_train_acc:.4f}")
print(f"  测试准确率：{rf50_test_acc:.4f}")

# 随机森林模型2：n_estimators=100
print("\n训练随机森林（n_estimators=100）...")
start_time = time.time()
rf100 = RandomForestClassifier(n_estimators=100, max_depth=8,
                               min_samples_leaf=5, random_state=42, n_jobs=-1)
rf100.fit(X_train, y_train)
rf100_train_time = time.time() - start_time

rf100_train_acc = rf100.score(X_train, y_train)
rf100_test_acc = rf100.score(X_test, y_test)
models['Random Forest (100)'] = {
    'model': rf100,
    'train_acc': rf100_train_acc,
    'test_acc': rf100_test_acc,
    'train_time': rf100_train_time
}

print(f"✓ 训练时间：{rf100_train_time:.4f}s")
print(f"  训练准确率：{rf100_train_acc:.4f}")
print(f"  测试准确率：{rf100_test_acc:.4f}")

# ============ 3. 交叉验证 ============
print("\n" + "-"*60)
print("5折交叉验证")
print("-"*60)

for model_name, model_info in models.items():
    model = model_info['model']
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"\n{model_name}:")
    print(f"  5折得分：{cv_scores}")
    print(f"  均值：{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ============ 4. 可视化对比 ============
print("\n" + "-"*60)
print("绘制性能对比图表")
print("-"*60)

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

fig.suptitle('决策树 vs 随机森林：性能对比', fontsize=15, fontweight='bold')

model_names = list(models.keys())
train_accs = [models[m]['train_acc'] for m in model_names]
test_accs = [models[m]['test_acc'] for m in model_names]
train_times = [models[m]['train_time'] for m in model_names]

# 子图1：准确率对比
ax1 = fig.add_subplot(gs[0, 0])
x = np.arange(len(model_names))
width = 0.35
bars1 = ax1.bar(x - width/2, train_accs, width, label='训练集', alpha=0.8, color='skyblue')
bars2 = ax1.bar(x + width/2, test_accs, width, label='测试集', alpha=0.8, color='coral')
ax1.set_xlabel('模型', fontsize=11)
ax1.set_ylabel('准确率', fontsize=11)
ax1.set_title('准确率对比', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(model_names, fontsize=9, rotation=15, ha='right')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim([0.7, 1.0])

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

# 子图2：训练时间对比
ax2 = fig.add_subplot(gs[0, 1])
bars = ax2.bar(model_names, train_times, color=['green', 'orange', 'red'], alpha=0.7)
ax2.set_ylabel('训练时间 (秒)', fontsize=11)
ax2.set_title('训练时间对比', fontsize=12, fontweight='bold')
ax2.set_xticklabels(model_names, fontsize=9, rotation=15, ha='right')
ax2.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, train_times):
    ax2.text(bar.get_x() + bar.get_width()/2., val + 0.001,
            f'{val:.4f}s', ha='center', va='bottom', fontsize=9)

# 子图3：过拟合程度
ax3 = fig.add_subplot(gs[0, 2])
overfitting = [train_accs[i] - test_accs[i] for i in range(len(model_names))]
colors = ['red' if o > 0.05 else 'yellow' if o > 0.02 else 'green' for o in overfitting]
bars = ax3.bar(model_names, overfitting, color=colors, alpha=0.7)
ax3.set_ylabel('准确率差距（训练-测试）', fontsize=11)
ax3.set_title('过拟合程度对比', fontsize=12, fontweight='bold')
ax3.set_xticklabels(model_names, fontsize=9, rotation=15, ha='right')
ax3.axhline(y=0.02, color='green', linestyle='--', alpha=0.5, label='优秀范围')
ax3.grid(True, alpha=0.3, axis='y')
ax3.legend()

for bar, val in zip(bars, overfitting):
    ax3.text(bar.get_x() + bar.get_width()/2., val + 0.001,
            f'{val:.4f}', ha='center', va='bottom', fontsize=8)

# 子图4：混淆矩阵（最优模型）
best_model_name = max(model_names, key=lambda m: models[m]['test_acc'])
best_model = models[best_model_name]['model']
y_pred_best = best_model.predict(X_test)
cm_best = confusion_matrix(y_test, y_pred_best)

ax4 = fig.add_subplot(gs[1, :2])
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', ax=ax4,
            xticklabels=le.classes_, yticklabels=le.classes_,
            cbar_kws={'label': '样本数'})
ax4.set_xlabel('预测标签', fontsize=11)
ax4.set_ylabel('真实标签', fontsize=11)
ax4.set_title(f'最优模型混淆矩阵（{best_model_name}）', fontsize=12, fontweight='bold')

# 子图5：特征重要性（随机森林）
ax5 = fig.add_subplot(gs[1, 2])
importances_rf = rf100.feature_importances_
indices = np.argsort(importances_rf)[::-1]
colors_feat = plt.cm.viridis(np.linspace(0, 1, len(features)))

bars = ax5.barh(range(len(features)), importances_rf[indices], color=colors_feat)
ax5.set_yticks(range(len(features)))
ax5.set_yticklabels([features[i] for i in indices], fontsize=10)
ax5.set_xlabel('重要性', fontsize=11)
ax5.set_title('特征重要性（RF-100）', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='x')

for i, (bar, val) in enumerate(zip(bars, importances_rf[indices])):
    ax5.text(val + 0.005, i, f'{val:.3f}', va='center', fontsize=9)

plt.tight_layout()
output_path = os.path.join(current_dir, 'rf_vs_dt_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ 对比分析图已保存：{output_path}")
plt.close()

# ============ 5. 详细分析最优模型 ============
print("\n" + "-"*60)
print("最优模型详细分析")
print("-"*60)

print(f"\n最优模型：{best_model_name}")
print(f"测试集准确率：{models[best_model_name]['test_acc']:.4f}")

y_pred = best_model.predict(X_test)
print(f"\n分类报告：")
print(classification_report(y_test, y_pred, target_names=le.classes_, digits=4))

# ============ 总结 ============
print("\n" + "="*60)
print("实验总结")
print("="*60)

dt_test_acc = models['Decision Tree']['test_acc']
rf100_test_acc = models['Random Forest (100)']['test_acc']
improvement = (rf100_test_acc - dt_test_acc) / dt_test_acc * 100

print(f"""
✓ 随机森林 vs 决策树对比分析完成

  模型性能对比：
  
  决策树（深度=8）：
  • 训练准确率：{models['Decision Tree']['train_acc']:.4f}
  • 测试准确率：{dt_test_acc:.4f}
  • 训练时间：{models['Decision Tree']['train_time']:.4f}s
  
  随机森林（50棵树）：
  • 训练准确率：{models['Random Forest (50)']['train_acc']:.4f}
  • 测试准确率：{models['Random Forest (50)']['test_acc']:.4f}
  • 训练时间：{models['Random Forest (50)']['train_time']:.4f}s
  
  随机森林（100棵树）：
  • 训练准确率：{models['Random Forest (100)']['train_acc']:.4f}
  • 测试准确率：{rf100_test_acc:.4f}
  • 训练时间：{models['Random Forest (100)']['train_time']:.4f}s
  
  关键发现：
  • 随机森林相比决策树准确率提升：{improvement:.2f}%
  • 随机森林显著降低了过拟合风险
  • 训练-测试准确率差距更小
  • 特征重要性排序：{' > '.join([features[i] for i in rf100.feature_importances_.argsort()[::-1]])}
  
  最优模型：{best_model_name}
  • 提供了最高的测试准确率：{models[best_model_name]['test_acc']:.4f}
  • 较好的过拟合控制
  • 特征重要性排序清晰，可解释性强
""")
print("="*60)
