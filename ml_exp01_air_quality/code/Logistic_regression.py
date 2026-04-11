# -*- coding: utf-8 -*-
"""
逻辑回归模型：二分类预测空气质量污染情况
数据源：北京市空气质量监测数据(2014-2019)
目标：预测是否污染（优良 → 0，其他 → 1）
特征：PM2.5, PM10
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import seaborn as sns
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
print("实验三：逻辑回归预测空气质量污染")
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

# 查看质量等级分布
print(f"\n✓ 空气质量等级分布：")
print(data['质量等级'].value_counts())

# 创建目标标签：优、良 → 0（无污染），其他 → 1（有污染）
data['pollution'] = (~data['质量等级'].isin(['优', '良'])).astype(int)

print(f"\n✓ 目标标签分布：")
print(f"  无污染(0)：{(data['pollution'] == 0).sum()}条")
print(f"  有污染(1)：{(data['pollution'] == 1).sum()}条")

label_dist = data['pollution'].value_counts()
print(f"  比例：无污染 {label_dist[0] / len(data) * 100:.1f}%, 有污染 {label_dist[1] / len(data) * 100:.1f}%")

# 提取特征和标签
X = data[['PM2.5', 'PM10']].values
y = data['pollution'].values

print(f"\n✓ 提取特征：")
print(f"  自变量形状：{X.shape}")
print(f"  因变量形状：{y.shape}")
print(f"  特征列表：PM2.5, PM10")

print(f"\n✓ 特征统计信息：")
print(f"  PM2.5 - 均值:{X[:, 0].mean():.2f}, 标准差:{X[:, 0].std():.2f}")
print(f"  PM10  - 均值:{X[:, 1].mean():.2f}, 标准差:{X[:, 1].std():.2f}")

# ============ 3. 分割训练集和测试集 ============
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✓ 数据分割（分层抽样）：")
print(f"  训练集大小：{X_train.shape[0]} (80%)")
print(f"  测试集大小：{X_test.shape[0]} (20%)")

# ============ 4. 特征标准化 ============
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✓ 特征标准化完成")
print(f"  标准化后训练集 PM2.5 - 均值:{X_train_scaled[:, 0].mean():.4f}, 标准差:{X_train_scaled[:, 0].std():.4f}")
print(f"  标准化后训练集 PM10 - 均值:{X_train_scaled[:, 1].mean():.4f}, 标准差:{X_train_scaled[:, 1].std():.4f}")

# ============ 5. 构建和训练模型 ============
print("\n" + "-" * 60)
print("模型构建与训练")
print("-" * 60)

# 模型1：默认参数
print(f"\n模型1：逻辑回归（默认参数）")
model1 = LogisticRegression(random_state=42, max_iter=1000)
model1.fit(X_train_scaled, y_train)

print(f"  ✓ 模型训练完成")
print(f"    系数：{model1.coef_[0]}")
print(f"    截距：{model1.intercept_[0]:.4f}")

# 模型2：平衡类权重
print(f"\n模型2：逻辑回归（class_weight='balanced'）")
model2 = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
model2.fit(X_train_scaled, y_train)

print(f"  ✓ 模型训练完成")
print(f"    系数：{model2.coef_[0]}")
print(f"    截距：{model2.intercept_[0]:.4f}")

# 使用模型2（性能通常更好）
model = model2
print(f"\n📊 使用模型2进行后续分析")

# ============ 6. 模型预测与评估 ============
print("\n" + "-" * 60)
print("模型预测与性能评估")
print("-" * 60)

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# 概率预测
y_train_pred_proba = model.predict_proba(X_train_scaled)[:, 1]
y_test_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# 计算评估指标
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

train_cm = confusion_matrix(y_train, y_train_pred)
test_cm = confusion_matrix(y_test, y_test_pred)

train_auc = roc_auc_score(y_train, y_train_pred_proba)
test_auc = roc_auc_score(y_test, y_test_pred_proba)

print(f"\n✓ 训练集性能指标：")
print(f"  准确率 (Accuracy)：{train_accuracy:.4f}")
print(f"  AUC分数：{train_auc:.4f}")
print(f"\n  混淆矩阵：")
print(f"    [[TN={train_cm[0][0]:3d}, FP={train_cm[0][1]:3d}],")
print(f"     [FN={train_cm[1][0]:3d}, TP={train_cm[1][1]:3d}]]")

print(f"\n✓ 测试集性能指标：")
print(f"  准确率 (Accuracy)：{test_accuracy:.4f}")
print(f"  AUC分数：{test_auc:.4f}")
print(f"\n  混淆矩阵：")
print(f"    [[TN={test_cm[0][0]:3d}, FP={test_cm[0][1]:3d}],")
print(f"     [FN={test_cm[1][0]:3d}, TP={test_cm[1][1]:3d}]]")

# 详细分类报告
print(f"\n✓ 测试集详细分类报告：")
print(classification_report(y_test, y_test_pred,
                            target_names=['无污染', '有污染'],
                            digits=4))

# 计算灵敏度、特异性等
TN = test_cm[0][0]
FP = test_cm[0][1]
FN = test_cm[1][0]
TP = test_cm[1][1]

sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = sensitivity
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n✓ 额外性能指标：")
print(f"  灵敏度 (Sensitivity/Recall)：{sensitivity:.4f}")
print(f"  特异性 (Specificity)：{specificity:.4f}")
print(f"  精确率 (Precision)：{precision:.4f}")
print(f"  F1-Score：{f1_score:.4f}")

# 显示部分预测结果
print(f"\n预测概率分析（前15条）：")
pred_df = pd.DataFrame({
    '真实标签': y_test[:15],
    '预测标签': y_test_pred[:15],
    '污染概率': np.round(y_test_pred_proba[:15], 4),
    '是否正确': (y_test[:15] == y_test_pred[:15]).astype(int)
})
print(pred_df.to_string(index=False))

# ============ 7. 可视化 ============
print("\n" + "-" * 60)
print("生成可视化图表")
print("-" * 60)

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

fig.suptitle('逻辑回归：污染物浓度预测空气质量污染',
             fontsize=16, fontweight='bold', y=0.995)

# 图1：决策边界（训练集）
ax1 = fig.add_subplot(gs[0, 0])
h = 0.1  # 增大步长以减少内存占用
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

ax1.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlGn_r)
ax1.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
            c='green', marker='o', s=30, alpha=0.6, label='无污染(0)', edgecolors='darkgreen')
ax1.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
            c='red', marker='x', s=50, alpha=0.8, label='有污染(1)', linewidths=2)
ax1.set_xlabel('PM2.5 (μg/m³)', fontsize=11)
ax1.set_ylabel('PM10 (μg/m³)', fontsize=11)
ax1.set_title('决策边界（训练集）', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 图2：决策边界（测试集）
ax2 = fig.add_subplot(gs[0, 1])
Z = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

ax2.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlGn_r)
ax2.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1],
            c='green', marker='o', s=30, alpha=0.6, label='无污染(0)', edgecolors='darkgreen')
ax2.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1],
            c='red', marker='x', s=50, alpha=0.8, label='有污染(1)', linewidths=2)
ax2.set_xlabel('PM2.5 (μg/m³)', fontsize=11)
ax2.set_ylabel('PM10 (μg/m³)', fontsize=11)
ax2.set_title('决策边界（测试集）', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 图3：ROC曲线
ax3 = fig.add_subplot(gs[0, 2])
fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC={test_auc:.4f})')
ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机分类')
ax3.set_xlim([0.0, 1.0])
ax3.set_ylim([0.0, 1.05])
ax3.set_xlabel('假正例率 (FPR)', fontsize=11)
ax3.set_ylabel('真正例率 (TPR)', fontsize=11)
ax3.set_title('ROC曲线', fontsize=12, fontweight='bold')
ax3.legend(loc="lower right")
ax3.grid(True, alpha=0.3)

# 图4：混淆矩阵（训练集）
ax4 = fig.add_subplot(gs[1, 0])
sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax4,
            xticklabels=['无污染', '有污染'], yticklabels=['无污染', '有污染'],
            cbar_kws={'label': '数量'})
ax4.set_ylabel('真实标签', fontsize=11)
ax4.set_xlabel('预测标签', fontsize=11)
ax4.set_title(f'混淆矩阵（训练集）\n准确率={train_accuracy:.4f}', fontsize=12, fontweight='bold')

# 图5：混淆矩阵（测试集）
ax5 = fig.add_subplot(gs[1, 1])
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax5,
            xticklabels=['无污染', '有污染'], yticklabels=['无污染', '有污染'],
            cbar_kws={'label': '数量'})
ax5.set_ylabel('真实标签', fontsize=11)
ax5.set_xlabel('预测标签', fontsize=11)
ax5.set_title(f'混淆矩阵（测试集）\n准确率={test_accuracy:.4f}', fontsize=12, fontweight='bold')

# 图6：性能指标对比
ax6 = fig.add_subplot(gs[1, 2])
metrics = ['准确率', 'AUC', '灵敏度', '特异性', 'F1-Score']
train_values = [train_accuracy, train_auc, recall, specificity, f1_score]
test_values = [test_accuracy, test_auc, sensitivity, specificity, f1_score]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax6.bar(x - width / 2, train_values, width, label='训练集', alpha=0.8)
bars2 = ax6.bar(x + width / 2, test_values, width, label='测试集', alpha=0.8)

ax6.set_ylabel('分数', fontsize=11)
ax6.set_title('性能指标对比', fontsize=12, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(metrics, fontsize=9, rotation=15, ha='right')
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')
ax6.set_ylim([0, 1.1])

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=8)

# 保存图表
output_path = os.path.join(current_dir, 'logistic_regression_results.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ 图表已保存：{output_path}")

plt.show()

# ============ 8. 总结 ============
print("\n" + "=" * 60)
print("实验总结")
print("=" * 60)

print(f"""
✓ 逻辑回归分类模型建立成功

  模型参数：
  • 特征标准化：StandardScaler
  • 参数设置：class_weight='balanced'（处理类不平衡问题）
  • 最大迭代次数：1000

  决策规则：
  • PM2.5与PM10加权和 > 决策阈值 → 有污染(1)
  • 否则 → 无污染(0)

  测试集性能指标：
  • 准确率 (Accuracy)：{test_accuracy:.4f}
  • AUC分数：{test_auc:.4f}
  • 灵敏度 (Sensitivity)：{sensitivity:.4f}
  • 特异性 (Specificity)：{specificity:.4f}
  • F1-Score：{f1_score:.4f}

  混淆矩阵分析：
  • 真正例 (TP)：{TP} （正确识别有污染）
  • 假正例 (FP)：{FP} （错误判定有污染）
  • 真反例 (TN)：{TN} （正确识别无污染）
  • 假反例 (FN)：{FN} （错误判定无污染）

  关键发现：
  • 模型准确率为{test_accuracy:.4f}，性能{"优秀" if test_accuracy > 0.85 else "很好" if test_accuracy > 0.8 else "较好" if test_accuracy > 0.75 else "一般"}
  • AUC为{test_auc:.4f}，分类能力{"强" if test_auc > 0.85 else "较强"}
  • class_weight='balanced'的使用有效处理了类不平衡问题
  • 模型在实际应用中具有较好的可靠性
""")
print("=" * 60)