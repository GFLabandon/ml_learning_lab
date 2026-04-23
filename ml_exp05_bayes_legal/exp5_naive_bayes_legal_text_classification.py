# -*- coding: utf-8 -*-
"""
实验五：基于朴素贝叶斯的法律文本分类
离婚诉讼文本的案情要素自动分类

目标：使用朴素贝叶斯分类器对法律文本进行多分类预测
类别：法律案情要素分类（案由、诉讼请求、事实与理由等）
数据：中文离婚诉讼文本（JSON格式）
"""

import json
import os
import re
import time
import warnings
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import jieba

warnings.filterwarnings('ignore')

# ============ 配置 ============
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("实验五：基于朴素贝叶斯的法律文本分类")
print("="*70)

# ============ 1. 数据加载与探索 ============
print("\n" + "-"*70)
print("第一步：数据加载与探索")
print("-"*70)

# 查找数据文件
data_paths = [
    '/mnt/user-data/uploads/离婚诉讼文本.json',
    './离婚诉讼文本.json',
    'divorce_cases.json',
    'legal_texts.json'
]

data_path = None
for path in data_paths:
    if os.path.exists(path):
        data_path = path
        print(f"✓ 找到数据文件：{data_path}")
        break

if data_path is None:
    print("\n⚠️  警告：未找到离婚诉讼文本.json数据文件")
    print("将使用模拟数据进行演示")
    # 创建模拟数据集
    data = {
        'texts': [
            "本案系离婚纠纷案件。原告与被告于2000年登记结婚，婚后感情逐渐破裂，经过多次调解无效，原告决定诉讼离婚。",
            "原告请求：一、判决原告与被告离婚；二、分割共同财产房产一套；三、孩子抚养权归原告。",
            "事实与理由：1. 夫妻感情已完全破裂；2. 被告长期外出，无故不归；3. 财产分割应当公平。",
            "经审理查明，原被告于2005年登记结婚，婚生子女一人。原告主张被告存在家暴行为。",
            "诉讼请求：1. 判决离婚；2. 被告赔偿精神损害费10万元；3. 子女由原告抚养。",
            "本案涉及离婚、财产分割、子女抚养三个主要问题。原告要求被告返还婚前个人财产。",
            "案件事实：被告与他人同居，经调解无效，感情完全破裂，满足离婚条件。",
            "诉讼请求明确：离婚、分财产、争取孩子抚养权、精神赔偿。",
        ],
        'labels': ['案由', '诉讼请求', '事实与理由', '案由', '诉讼请求', '案由', '法院查明', '诉讼请求']
    }
else:
    def normalize_dataset(data_raw):
        """将原始JSON数据统一转换为 {'texts': [...], 'labels': [...]} 格式。"""
        data_norm = {'texts': [], 'labels': []}

        if isinstance(data_raw, list):
            for item in data_raw:
                if isinstance(item, dict):
                    text = (
                            item.get('text') or
                            item.get('content') or
                            item.get('文本') or
                            item.get('sentence')  # ✅ 加这一行
                    )

                    label = (
                            item.get('label') or
                            item.get('category') or
                            item.get('标签') or
                            (item.get('labels')[0] if item.get('labels') else '未分类')  # ✅ 适配你的数据
                    )
                    if text:
                        data_norm['texts'].append(str(text))
                        data_norm['labels'].append(str(label))
        elif isinstance(data_raw, dict):
            if 'texts' in data_raw and 'labels' in data_raw:
                texts = data_raw.get('texts', [])
                labels = data_raw.get('labels', [])
                if isinstance(texts, list) and isinstance(labels, list):
                    paired_len = min(len(texts), len(labels))
                    data_norm['texts'] = [str(t) for t in texts[:paired_len]]
                    data_norm['labels'] = [str(l) for l in labels[:paired_len]]
            else:
                for _, value in data_raw.items():
                    if isinstance(value, dict):
                        text = value.get('text') or value.get('content') or value.get('文本')
                        label = value.get('label') or value.get('category') or value.get('标签') or '未分类'
                        if text:
                            data_norm['texts'].append(str(text))
                            data_norm['labels'].append(str(label))

        return data_norm

    # 加载真实数据（优先标准 JSON，失败后尝试 JSON Lines）
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data_raw = json.load(f)
        data = normalize_dataset(data_raw)
        if not data['texts']:
            raise ValueError("标准 JSON 解析成功，但未提取到有效文本样本。")
        print(f"✓ 数据加载成功")
    except json.JSONDecodeError:
        print("⚠️  标准 JSON 解析失败，尝试按 JSON Lines 格式加载...")
        try:
            data_raw = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data_raw.append(json.loads(line))
            data = normalize_dataset(data_raw)
            if not data['texts']:
                raise ValueError("JSON Lines 解析成功，但未提取到有效文本样本。")
            print("✓ JSON Lines 数据加载成功")
        except Exception as e:
            print(f"✗ 加载数据失败：{e}")
            print("使用模拟数据进行演示")
            data = {
                'texts': [
                    "本案系离婚纠纷案件。原告与被告于2000年登记结婚，婚后感情逐渐破裂，经过多次调解无效，原告决定诉讼离婚。",
                    "再次强调，本案属于典型的离婚纠纷，双方长期分居，调解后仍无法和好。",
                    "原告请求：一、判决原告与被告离婚；二、分割共同财产房产一套；三、孩子抚养权归原告。",
                    "诉讼请求包括依法判决离婚、确认抚养权归属并公平分割夫妻共同财产。",
                    "事实与理由：1. 夫妻感情已完全破裂；2. 被告长期外出，无故不归；3. 财产分割应当公平。",
                    "事实与理由补充：被告存在家庭暴力行为，导致婚姻关系难以继续。",
                ],
                'labels': ['案由', '案由', '诉讼请求', '诉讼请求', '事实与理由', '事实与理由']
            }
    except Exception as e:
        print(f"✗ 加载数据失败：{e}")
        print("使用模拟数据进行演示")
        data = {
            'texts': [
                "本案系离婚纠纷案件。原告与被告于2000年登记结婚，婚后感情逐渐破裂，经过多次调解无效，原告决定诉讼离婚。",
                "再次强调，本案属于典型的离婚纠纷，双方长期分居，调解后仍无法和好。",
                "原告请求：一、判决原告与被告离婚；二、分割共同财产房产一套；三、孩子抚养权归原告。",
                "诉讼请求包括依法判决离婚、确认抚养权归属并公平分割夫妻共同财产。",
                "事实与理由：1. 夫妻感情已完全破裂；2. 被告长期外出，无故不归；3. 财产分割应当公平。",
                "事实与理由补充：被告存在家庭暴力行为，导致婚姻关系难以继续。",
            ],
            'labels': ['案由', '案由', '诉讼请求', '诉讼请求', '事实与理由', '事实与理由']
        }

# 数据预处理
print(f"\n✓ 数据集信息：")
print(f"  文本数：{len(data['texts'])}")
print(f"  类别数：{len(set(data['labels']))}")
print(f"  类别分布：{dict(Counter(data['labels']))}")

# ============ 2. 文本预处理 ============
print("\n" + "-"*70)
print("第二步：文本预处理")
print("-"*70)

def clean_text(text):
    """文本清洗"""
    if not isinstance(text, str):
        return ""
    # 移除特殊字符，保留中文和英文
    text = re.sub(r'[^\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ffa-zA-Z0-9]', '', text)
    return text

def segment_text(text):
    """中文分词"""
    # 使用jieba进行中文分词
    try:
        words = jieba.cut(text)
        return list(words)
    except:
        # 如果jieba不可用，按字符分割
        return list(text)

def remove_stopwords(words, stopwords=None):
    """移除停用词"""
    if stopwords is None:
        # 默认停用词列表
        stopwords = {'的', '了', '和', '是', '在', '有', '等', '被', '也', '这'}
    return [w for w in words if w not in stopwords and len(w) > 0]

# 清洗和分词
print(f"✓ 进行文本清洗和分词...")
processed_texts = []
for text in data['texts']:
    # 清洗
    cleaned = clean_text(text)
    # 分词
    words = segment_text(cleaned)
    # 移除停用词
    words = remove_stopwords(words)
    # 合并为字符串（用空格分隔）
    processed_text = ' '.join(words)
    processed_texts.append(processed_text)

print(f"✓ 文本预处理完成")
print(f"  样本1（处理前）：{data['texts'][0][:50]}...")
print(f"  样本1（处理后）：{processed_texts[0][:80]}...")

# ============ 3. 特征提取 ============
print("\n" + "-"*70)
print("第三步：特征提取")
print("-"*70)

# 方法1：词袋模型（BoW）
print(f"\n✓ 特征提取方法1：词袋模型（BoW）")
bow_vectorizer = CountVectorizer(max_features=100, min_df=1, max_df=0.9)
X_bow = bow_vectorizer.fit_transform(processed_texts)
print(f"  特征维度：{X_bow.shape}")
print(f"  样本数：{X_bow.shape[0]}")
print(f"  特征数：{X_bow.shape[1]}")

# 方法2：TF-IDF
print(f"\n✓ 特征提取方法2：TF-IDF")
tfidf_vectorizer = TfidfVectorizer(max_features=100, min_df=1, max_df=0.9)
X_tfidf = tfidf_vectorizer.fit_transform(processed_texts)
print(f"  特征维度：{X_tfidf.shape}")

# 编码标签
print(f"\n✓ 标签编码")
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['labels'])
print(f"  类别映射：{dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

# ============ 4. 数据划分 ============
print("\n" + "-"*70)
print("第四步：数据划分")
print("-"*70)

# 划分为训练集和测试集（80%-20%）
label_counts = Counter(y)
min_class_count = min(label_counts.values()) if label_counts else 0
num_classes = len(label_counts)
test_size = 0.2
test_count = int(np.ceil(len(y) * test_size))

use_stratify = (min_class_count >= 2) and (test_count >= num_classes)
stratify_target = y if use_stratify else None

if not use_stratify:
    print("⚠️  当前数据规模/类别分布不满足分层抽样条件，已自动切换为普通随机划分。")

X_bow_train, X_bow_test, y_train, y_test = train_test_split(
    X_bow, y, test_size=test_size, random_state=42, stratify=stratify_target
)
X_tfidf_train, X_tfidf_test, _, _ = train_test_split(
    X_tfidf, y, test_size=test_size, random_state=42, stratify=stratify_target
)

print(f"✓ 数据划分完成：")
print(f"  训练集：{X_bow_train.shape[0]} 样本")
print(f"  测试集：{X_bow_test.shape[0]} 样本")
train_class_distribution = {label_encoder.inverse_transform([k])[0]: v for k, v in Counter(y_train).items()}
print(f"  类别分布（训练集）：{train_class_distribution}")

# ============ 5. 模型训练与评估 ============
print("\n" + "-"*70)
print("第五步：模型训练与评估")
print("-"*70)

results = {}

# 模型1：多项式朴素贝叶斯（BoW特征）
print(f"\n【模型1】多项式朴素贝叶斯 + 词袋模型")
start_time = time.time()
mnb_bow = MultinomialNB(alpha=1.0)
mnb_bow.fit(X_bow_train, y_train)
train_time_bow = time.time() - start_time

y_train_pred_bow = mnb_bow.predict(X_bow_train)
y_test_pred_bow = mnb_bow.predict(X_bow_test)

train_acc_bow = accuracy_score(y_train, y_train_pred_bow)
test_acc_bow = accuracy_score(y_test, y_test_pred_bow)

results['MNB_BoW'] = {
    'model': mnb_bow,
    'X_train': X_bow_train,
    'X_test': X_bow_test,
    'y_train_pred': y_train_pred_bow,
    'y_test_pred': y_test_pred_bow,
    'train_acc': train_acc_bow,
    'test_acc': test_acc_bow,
    'train_time': train_time_bow,
}

print(f"  训练准确率：{train_acc_bow:.4f}")
print(f"  测试准确率：{test_acc_bow:.4f}")
print(f"  训练时间：{train_time_bow:.4f}s")

# 模型2：多项式朴素贝叶斯（TF-IDF特征）
print(f"\n【模型2】多项式朴素贝叶斯 + TF-IDF")
start_time = time.time()
mnb_tfidf = MultinomialNB(alpha=1.0)
mnb_tfidf.fit(X_tfidf_train, y_train)
train_time_tfidf = time.time() - start_time

y_train_pred_tfidf = mnb_tfidf.predict(X_tfidf_train)
y_test_pred_tfidf = mnb_tfidf.predict(X_tfidf_test)

train_acc_tfidf = accuracy_score(y_train, y_train_pred_tfidf)
test_acc_tfidf = accuracy_score(y_test, y_test_pred_tfidf)

results['MNB_TF-IDF'] = {
    'model': mnb_tfidf,
    'X_train': X_tfidf_train,
    'X_test': X_tfidf_test,
    'y_train_pred': y_train_pred_tfidf,
    'y_test_pred': y_test_pred_tfidf,
    'train_acc': train_acc_tfidf,
    'test_acc': test_acc_tfidf,
    'train_time': train_time_tfidf,
}

print(f"  训练准确率：{train_acc_tfidf:.4f}")
print(f"  测试准确率：{test_acc_tfidf:.4f}")
print(f"  训练时间：{train_time_tfidf:.4f}s")

# ============ 6. 最优模型分析 ============
print("\n" + "-"*70)
print("第六步：最优模型分析")
print("-"*70)

best_model_key = max(results.keys(), key=lambda k: results[k]['test_acc'])
best_result = results[best_model_key]

print(f"\n✓ 最优模型：{best_model_key}")
print(f"  训练准确率：{best_result['train_acc']:.4f}")
print(f"  测试准确率：{best_result['test_acc']:.4f}")
print(f"  训练时间：{best_result['train_time']:.4f}s")

# ============ 7. 详细评估 ============
print("\n" + "-"*70)
print("第七步：详细评估")
print("-"*70)

y_test_pred = best_result['y_test_pred']

# 混淆矩阵
cm = confusion_matrix(
    y_test,
    y_test_pred,
    labels=np.arange(len(label_encoder.classes_))
)

print(f"\n✓ 分类报告（测试集）：")
print(classification_report(
    y_test,
    y_test_pred,
    labels=np.arange(len(label_encoder.classes_)),
    target_names=label_encoder.classes_,
    digits=4,
    zero_division=0
))

# 计算指标
precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

print(f"\n✓ 加权平均指标：")
print(f"  精确率（Precision）：{precision:.4f}")
print(f"  召回率（Recall）：{recall:.4f}")
print(f"  F1分数：{f1:.4f}")

# ============ 8. 交叉验证 ============
print("\n" + "-"*70)
print("第八步：交叉验证")
print("-"*70)

# 自适应交叉验证折数（避免小样本报错）
train_min_class_count = min(Counter(y_train).values()) if len(y_train) > 0 else 0
cv_folds = min(5, train_min_class_count)

if cv_folds >= 2:
    cv_scores = cross_val_score(
        MultinomialNB(alpha=1.0),
        best_result['X_train'],
        y_train,
        cv=cv_folds,
        scoring='accuracy'
    )
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    print(f"✓ {cv_folds}折交叉验证结果：")
else:
    cv_scores = np.array([best_result['test_acc']])
    cv_mean = cv_scores.mean()
    cv_std = 0.0
    print("⚠️  训练集中每类样本不足2个，无法执行交叉验证，使用测试集准确率作为参考值。")

print(f"  各折准确率：{[f'{score:.4f}' for score in cv_scores]}")
print(f"  平均准确率：{cv_mean:.4f}")
print(f"  标准差：{cv_std:.4f}")

# ============ 9. 创建结果对比表 ============
print("\n" + "-"*70)
print("第九步：结果总结与对比")
print("-"*70)

comparison_data = []
for key, result in results.items():
    comparison_data.append({
        '模型': key,
        '训练准确率': f"{result['train_acc']:.4f}",
        '测试准确率': f"{result['test_acc']:.4f}",
        '训练时间(s)': f"{result['train_time']:.4f}",
        '过拟合度': f"{(result['train_acc'] - result['test_acc']):.4f}"
    })

df_results = pd.DataFrame(comparison_data)
print("\n" + df_results.to_string(index=False))

# ============ 10. 结果可视化 ============
print("\n" + "-"*70)
print("第十步：结果可视化")
print("-"*70)

# 准备绘图数据
model_names = list(results.keys())
train_accs = [results[m]['train_acc'] for m in model_names]
test_accs = [results[m]['test_acc'] for m in model_names]
train_times = [results[m]['train_time'] for m in model_names]

# 绘制4个子图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('朴素贝叶斯文本分类性能对比', fontsize=14, fontweight='bold')

# 子图1：训练和测试准确率对比
ax1 = axes[0, 0]
x = np.arange(len(model_names))
width = 0.35

bars1 = ax1.bar(x - width/2, train_accs, width, label='训练准确率', alpha=0.8, color='steelblue')
bars2 = ax1.bar(x + width/2, test_accs, width, label='测试准确率', alpha=0.8, color='coral')

ax1.set_xlabel('模型', fontsize=11)
ax1.set_ylabel('准确率', fontsize=11)
ax1.set_title('训练与测试准确率对比', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(model_names, rotation=15, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim([0.0, 1.0])

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# 子图2：训练时间对比
ax2 = axes[0, 1]
bars = ax2.bar(model_names, train_times, color='mediumseagreen', alpha=0.8)

ax2.set_xlabel('模型', fontsize=11)
ax2.set_ylabel('训练时间（秒）', fontsize=11)
ax2.set_title('模型训练时间对比', fontsize=12, fontweight='bold')
ax2.set_xticklabels(model_names, rotation=15, ha='right')
ax2.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}s', ha='center', va='bottom', fontsize=9)

# 子图3：过拟合度分析
ax3 = axes[1, 0]
overfitting = [train_accs[i] - test_accs[i] for i in range(len(model_names))]
colors_of = ['red' if o > 0.1 else 'green' for o in overfitting]
bars = ax3.bar(model_names, overfitting, color=colors_of, alpha=0.7)

ax3.set_xlabel('模型', fontsize=11)
ax3.set_ylabel('过拟合度（训练-测试准确率）', fontsize=11)
ax3.set_title('过拟合度分析', fontsize=12, fontweight='bold')
ax3.set_xticklabels(model_names, rotation=15, ha='right')
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax3.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

# 子图4：混淆矩阵热力图
ax4 = axes[1, 1]
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_normalized = np.nan_to_num(cm_normalized)

if len(label_encoder.classes_) <= 10:
    im = ax4.imshow(cm_normalized, cmap='Blues', aspect='auto')
    ax4.set_xticks(np.arange(len(label_encoder.classes_)))
    ax4.set_yticks(np.arange(len(label_encoder.classes_)))
    ax4.set_xticklabels(label_encoder.classes_, rotation=45, ha='right', fontsize=9)
    ax4.set_yticklabels(label_encoder.classes_, fontsize=9)
    
    # 添加数值
    num_classes = cm.shape[0]

    for i in range(num_classes):
        for j in range(num_classes):
            ax4.text(j, i, f'{cm_normalized[i, j]:.2f}',
                     ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=ax4)
else:
    # 类别太多时显示总体统计
    accuracy_per_class = np.diag(cm) / cm.sum(axis=1)
    ax4.barh(label_encoder.classes_, accuracy_per_class)

ax4.set_xlabel('预测标签', fontsize=11)
ax4.set_ylabel('真实标签', fontsize=11)
ax4.set_title('混淆矩阵（归一化）', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('naive_bayes_text_classification.png', dpi=150, bbox_inches='tight')
print(f"✓ 结果图表已保存：naive_bayes_text_classification.png")
plt.show()

# ============ 11. 特征分析 ============
print("\n" + "-"*70)
print("第十一步：特征重要性分析")
print("-"*70)

best_model = best_result['model']

# 获取最重要的特征（词）
feature_names = np.array(bow_vectorizer.get_feature_names_out() if hasattr(bow_vectorizer, 'get_feature_names_out') 
                         else bow_vectorizer.get_feature_names())

# 计算每个类别最重要的特征
for i, class_label in enumerate(label_encoder.classes_):
    if hasattr(best_model, 'feature_log_prob_'):
        # 获取该类别的特征权重
        feature_weights = best_model.feature_log_prob_[i]
        top_indices = np.argsort(feature_weights)[-5:][::-1]
        top_features = feature_names[top_indices]
        
        print(f"\n✓ 类别『{class_label}』的核心特征词：")
        for j, feature in enumerate(top_features, 1):
            print(f"  {j}. {feature}")

# ============ 12. 分析与结论 ============
print("\n" + "-"*70)
print("第十二步：分析与结论")
print("-"*70)

print(f"""
✓ 实验结果总结：

【模型性能】
  最优模型：{best_model_key}
  测试准确率：{best_result['test_acc']:.4f}
  精确率：{precision:.4f}
  召回率：{recall:.4f}
  F1分数：{f1:.4f}
  
【特征对比】
  词袋模型：简单快速，特征数{X_bow.shape[1]}
  TF-IDF：考虑词频，特征数{X_tfidf.shape[1]}
  
【朴素贝叶斯特性】
  • 假设：特征条件独立
  • 优点：训练速度快，数据需求少
  • 缺点：对特征独立性依赖强
  • 适用：文本分类、情感分析、垃圾邮件过滤
  
【模型表现分析】
  • 过拟合度：{(train_accs[0] - test_accs[0]):.4f}
  • 交叉验证平均准确率：{cv_mean:.4f}
  • 模型稳定性：{'良好' if cv_std < 0.05 else '需改进'}
  
【改进建议】
  1. 增加训练样本量，特别是少数类样本
  2. 优化停用词表和分词方法
  3. 尝试不同的特征权重方案（TF-IDF权重调整）
  4. 使用集成方法（投票、堆叠等）
  5. 进行超参数调优（alpha参数）
  6. 考虑类别不平衡问题（class_weight参数）

【法律文本分类应用】
  • 自动识别法律文本中的关键要素
  • 加快案件信息录入和分析速度
  • 支持智能检索和知识管理
  • 辅助法官和律师的工作效率
""")

print("\n" + "="*70)
print("实验完成！")
print("="*70)
