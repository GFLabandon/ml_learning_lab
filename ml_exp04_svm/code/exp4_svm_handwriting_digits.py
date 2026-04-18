# -*- coding: utf-8 -*-
"""
实验四：基于SVM的手写数字识别
使用MNIST手写数据集，对比不同核函数和多分类策略

目标：实现0-9手写数字的10分类预测
核函数：linear, poly, rbf, sigmoid
多分类策略：OVO (One-vs-One), OVR (One-vs-Rest)
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pickle
import os
import time
import warnings
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import seaborn as sns

warnings.filterwarnings('ignore')

# ── 0. 中文字体配置（跨平台） ─────────────────────────────────────────────
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = [
    'SimHei', 'STHeiti', 'Noto Sans CJK SC', 'Noto Sans CJK JP',
    'WenQuanYi Zen Hei', 'Arial Unicode MS', 'DejaVu Sans'
]

print("="*70)
print("实验四：基于SVM的手写数字识别")
print("="*70)

# ============ 1. 数据加载与预处理 ============
print("\n" + "-"*70)
print("第一步：数据加载与预处理")
print("-"*70)

# 查找mnist.pkl文件
data_paths = [
    '/mnt/user-data/uploads/mnist.pkl.gz',
    '/mnt/user-data/uploads/mnist.pkl',
    './mnist.pkl.gz',
    './mnist.pkl',
    'mnist.pkl.gz',
    'mnist.pkl'
]

data_path = None
for path in data_paths:
    if os.path.exists(path):
        data_path = path
        print(f"✓ 找到数据文件：{data_path}")
        break

if data_path is None:
    print("\n⚠️  警告：未找到mnist.pkl数据文件")
    print("将使用scikit-learn中的digits数据集作为演示")
    from sklearn.datasets import load_digits
    data = load_digits()
    X = data.data
    y = data.target
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    print(f"✓ 使用digits数据集（8x8像素的手写数字0-9）")
else:
    # 加载pkl数据
    try:
        import gzip
        if data_path.endswith('.gz'):
            with gzip.open(data_path, 'rb') as f:
                train_data, test_data = pickle.load(f, encoding='bytes')
        else:
            with open(data_path, 'rb') as f:
                train_data, test_data = pickle.load(f, encoding='bytes')
        
        X_train, y_train = train_data[0], train_data[1]
        X_test, y_test = test_data[0], test_data[1]
        X = np.vstack([X_train, X_test])
        y = np.concatenate([y_train, y_test])
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        print(f"✓ 数据加载成功")
    except Exception as e:
        print(f"✗ 加载mnist.pkl失败：{e}")
        print("改用scikit-learn的digits数据集")
        from sklearn.datasets import load_digits
        data = load_digits()
        X = data.data
        y = data.target
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

# 数据信息
print(f"✓ 数据集信息：")
print(f"  样本数：{n_samples}")
print(f"  特征数：{n_features}")
print(f"  类别数：{n_classes}")
print(f"  类别分布：{np.bincount(y)}")

# 归一化（0-1缩放）
print(f"\n✓ 数据归一化（0-1缩放）...")
X_min, X_max = X.min(), X.max()
X = (X - X_min) / (X_max - X_min)
print(f"  数据范围：[{X.min():.4f}, {X.max():.4f}]")

# 二值化（可选，取决于数据特性）
# X = (X > 0.5).astype(int)

# 划分训练/验证/测试集
print(f"\n✓ 数据划分...")
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
)

print(f"  训练集：{len(X_train)} 样本")
print(f"  验证集：{len(X_val)} 样本")
print(f"  测试集：{len(X_test)} 样本")

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
print(f"✓ StandardScaler标准化完成")

# ============ 2. 对比不同核函数和多分类策略 ============
print("\n" + "-"*70)
print("第二步：对比不同核函数和多分类策略")
print("-"*70)

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
strategies = ['ovr', 'ovo']

results = {}
kernel_colors = {'linear': '#1f77b4', 'poly': '#ff7f0e', 'rbf': '#2ca02c', 'sigmoid': '#d62728'}

print(f"\n{'核函数':<12} {'策略':<6} {'训练准确率':<12} {'验证准确率':<12} {'测试准确率':<12} {'训练时间':<12}")
print("-"*70)

for kernel in kernels:
    for strategy in strategies:
        print(f"{kernel:<12} {strategy:<6}", end="", flush=True)
        
        # 设置SVM参数
        svm_params = {
            'kernel': kernel,
            'C': 1.0,
            'decision_function_shape': strategy,
            'random_state': 42
        }
        
        # 针对poly核函数添加degree参数
        if kernel == 'poly':
            svm_params['degree'] = 3
        
        # 针对rbf和sigmoid核函数的gamma参数
        if kernel in ['rbf', 'sigmoid']:
            svm_params['gamma'] = 'scale'
        
        # 训练
        start_time = time.time()
        svm = SVC(**svm_params)
        svm.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # 预测
        y_train_pred = svm.predict(X_train)
        y_val_pred = svm.predict(X_val)
        y_test_pred = svm.predict(X_test)
        
        # 评估
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        # 保存结果
        key = f"{kernel}_{strategy}"
        results[key] = {
            'svm': svm,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'train_time': train_time,
            'y_test_pred': y_test_pred,
            'y_train_pred': y_train_pred
        }
        
        print(f" {train_acc:.4f}       {val_acc:.4f}       {test_acc:.4f}       {train_time:.4f}s")

# ============ 3. 分析最优模型 ============
print("\n" + "-"*70)
print("第三步：最优模型分析")
print("-"*70)

# 找出测试集准确率最高的模型
best_model_key = max(results.keys(), key=lambda k: results[k]['test_acc'])
best_kernel, best_strategy = best_model_key.split('_')

print(f"\n✓ 最优模型：")
print(f"  核函数：{best_kernel}")
print(f"  多分类策略：{best_strategy}")
print(f"  训练准确率：{results[best_model_key]['train_acc']:.4f}")
print(f"  验证准确率：{results[best_model_key]['val_acc']:.4f}")
print(f"  测试准确率：{results[best_model_key]['test_acc']:.4f}")
print(f"  训练时间：{results[best_model_key]['train_time']:.4f}s")

# ============ 4. 最优模型详细评估 ============
print("\n" + "-"*70)
print("第四步：最优模型详细评估")
print("-"*70)

best_svm = results[best_model_key]['svm']
y_test_pred = results[best_model_key]['y_test_pred']

# 混淆矩阵
cm = confusion_matrix(y_test, y_test_pred)

print(f"\n✓ 分类报告（测试集）：")
print(classification_report(y_test, y_test_pred, digits=4))

# 计算精确率、召回率、F1分数
precision = precision_score(y_test, y_test_pred, average='weighted')
recall = recall_score(y_test, y_test_pred, average='weighted')
f1 = f1_score(y_test, y_test_pred, average='weighted')

print(f"\n✓ 加权平均指标：")
print(f"  精确率（Precision）：{precision:.4f}")
print(f"  召回率（Recall）：{recall:.4f}")
print(f"  F1分数：{f1:.4f}")

# ============ 5. 创建结果对比表 ============
print("\n" + "-"*70)
print("第五步：结果总结与对比")
print("-"*70)

# 创建对比表格
comparison_data = []
for key, result in results.items():
    kernel, strategy = key.split('_')
    comparison_data.append({
        '核函数': kernel,
        '多分类策略': strategy,
        '训练准确率': f"{result['train_acc']:.4f}",
        '验证准确率': f"{result['val_acc']:.4f}",
        '测试准确率': f"{result['test_acc']:.4f}",
        '训练时间(s)': f"{result['train_time']:.4f}"
    })

df_results = pd.DataFrame(comparison_data)
print("\n" + df_results.to_string(index=False))

# ============ 6. 结果分析与可视化 ============
print("\n" + "-"*70)
print("第六步：结果可视化")
print("-"*70)

# 准备数据用于绘图
kernel_list = []
ovr_test_accs = []
ovo_test_accs = []
ovr_times = []
ovo_times = []

for kernel in kernels:
    kernel_list.append(kernel)
    ovr_key = f"{kernel}_ovr"
    ovo_key = f"{kernel}_ovo"
    
    ovr_test_accs.append(results[ovr_key]['test_acc'])
    ovo_test_accs.append(results[ovo_key]['test_acc'])
    ovr_times.append(results[ovr_key]['train_time'])
    ovo_times.append(results[ovo_key]['train_time'])

# 绘制4个子图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('SVM手写数字识别：核函数与多分类策略对比', fontsize=14, fontweight='bold')

# 子图1：测试准确率对比（核函数）
ax1 = axes[0, 0]
x = np.arange(len(kernel_list))
width = 0.35

bars1 = ax1.bar(x - width/2, ovr_test_accs, width, label='OVR', alpha=0.8, color='steelblue')
bars2 = ax1.bar(x + width/2, ovo_test_accs, width, label='OVO', alpha=0.8, color='coral')

ax1.set_xlabel('核函数', fontsize=11)
ax1.set_ylabel('测试准确率', fontsize=11)
ax1.set_title('不同核函数的测试准确率对比', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(kernel_list)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim([0.8, 1.0])

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

# 子图2：训练时间对比（核函数）
ax2 = axes[0, 1]
bars1 = ax2.bar(x - width/2, ovr_times, width, label='OVR', alpha=0.8, color='steelblue')
bars2 = ax2.bar(x + width/2, ovo_times, width, label='OVO', alpha=0.8, color='coral')

ax2.set_xlabel('核函数', fontsize=11)
ax2.set_ylabel('训练时间（秒）', fontsize=11)
ax2.set_title('不同核函数的训练时间对比', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(kernel_list)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

# 子图3：所有模型的测试准确率（按核函数分组）
ax3 = axes[1, 0]
all_models = []
all_accs = []
colors = []

for kernel in kernels:
    for strategy in strategies:
        key = f"{kernel}_{strategy}"
        all_models.append(f"{kernel}\n{strategy}")
        all_accs.append(results[key]['test_acc'])
        colors.append(kernel_colors[kernel])

x_pos = np.arange(len(all_models))
bars = ax3.bar(x_pos, all_accs, color=colors, alpha=0.7)
ax3.set_xlabel('核函数-策略组合', fontsize=11)
ax3.set_ylabel('测试准确率', fontsize=11)
ax3.set_title('所有模型组合的测试准确率', fontsize=12, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(all_models, fontsize=8, rotation=45, ha='right')
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_ylim([0.8, 1.0])

# 标记最优模型
best_idx = all_models.index(f"{best_kernel}\n{best_strategy}")
bars[best_idx].set_edgecolor('red')
bars[best_idx].set_linewidth(3)

# 子图4：混淆矩阵热力图
ax4 = axes[1, 1]
# 对于大规模混淆矩阵，采用缩小显示
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# 如果类别数过多，显示精简版
if n_classes <= 10:
    sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues', ax=ax4, cbar=True)
    ax4.set_xlabel('预测标签', fontsize=11)
    ax4.set_ylabel('真实标签', fontsize=11)
else:
    # 对于大规模数据，显示总体统计
    ax4.text(0.5, 0.5, f'混淆矩阵\n尺寸: {n_classes}×{n_classes}\n\n测试准确率: {results[best_model_key]["test_acc"]:.4f}',
            horizontalalignment='center', verticalalignment='center',
            transform=ax4.transAxes, fontsize=12, fontweight='bold')
    ax4.axis('off')

ax4.set_title(f'混淆矩阵（最优模型）', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('svm_handwriting_digits_analysis.png', dpi=150, bbox_inches='tight')
print(f"✓ 结果图表已保存：svm_handwriting_digits_analysis.png")
plt.show()

# ============ 7. 详细分析 ============
print("\n" + "-"*70)
print("第七步：详细分析与结论")
print("-"*70)

print(f"""
✓ 核函数对比分析：

1. Linear（线性核）：
   - 适用于线性可分的数据
   - 训练速度快，参数少
   - 本数据集表现：OVR准确率 {results['linear_ovr']['test_acc']:.4f}，OVO准确率 {results['linear_ovo']['test_acc']:.4f}

2. Polynomial（多项式核）：
   - 可以处理高维非线性问题
   - 计算成本较高，需要调整degree参数
   - 本数据集表现：OVR准确率 {results['poly_ovr']['test_acc']:.4f}，OVO准确率 {results['poly_ovo']['test_acc']:.4f}

3. RBF（径向基函数核）：
   - 最灵活，适合大多数非线性问题
   - 需要调整gamma参数
   - 本数据集表现：OVR准确率 {results['rbf_ovr']['test_acc']:.4f}，OVO准确率 {results['rbf_ovo']['test_acc']:.4f}

4. Sigmoid（sigmoid核）：
   - 与神经网络相似
   - 计算成本高，有时欠稳定
   - 本数据集表现：OVR准确率 {results['sigmoid_ovr']['test_acc']:.4f}，OVO准确率 {results['sigmoid_ovo']['test_acc']:.4f}

✓ 多分类策略对比分析：

1. OVR（One-vs-Rest）：
   - 为每个类别训练一个二分类器（k个分类器）
   - 训练速度较快
   - 平均准确率：{np.mean([results[f'{k}_ovr']['test_acc'] for k in kernels]):.4f}
   - 平均训练时间：{np.mean([results[f'{k}_ovr']['train_time'] for k in kernels]):.4f}s

2. OVO（One-vs-One）：
   - 为每对类别训练一个二分类器（k(k-1)/2个分类器）
   - 训练复杂度高，但每个分类器相对简单
   - 平均准确率：{np.mean([results[f'{k}_ovo']['test_acc'] for k in kernels]):.4f}
   - 平均训练时间：{np.mean([results[f'{k}_ovo']['train_time'] for k in kernels]):.4f}s

✓ 最优模型：
  核函数：{best_kernel.upper()}
  多分类策略：{best_strategy.upper()}
  测试准确率：{results[best_model_key]['test_acc']:.4f}
  精确率：{precision:.4f}
  召回率：{recall:.4f}
  F1分数：{f1:.4f}
  训练时间：{results[best_model_key]['train_time']:.4f}秒

✓ 建议：
  • 对于手写数字识别任务，{best_kernel.upper()}核函数表现最优
  • {best_strategy.upper()}多分类策略在准确率和速度间实现了更好的平衡
  • 进一步改进可以考虑：
    - 调整SVM参数（C, gamma）
    - 尝试特征工程（PCA降维、特征缩放等）
    - 使用集成方法结合多个分类器
    - 应用数据增强提升样本量
""")

print("\n" + "="*70)
print("实验完成！")
print("="*70)
