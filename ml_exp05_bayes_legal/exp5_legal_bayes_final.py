# -*- coding: utf-8 -*-
"""
实验五：法律裁判文书中的案情要素贝叶斯分类
数据集：离婚诉讼文本_part.json（JSON Lines格式，每行是一个JSON数组）
模型：TF-IDF + 多项式朴素贝叶斯（MultinomialNB）
作者：8组（郭飞龙、王泽凯、陈浩天、薛子琛）
"""

import json
import os
import re
import time
import warnings
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

import jieba

warnings.filterwarnings('ignore')

# ─── 跨平台中文字体配置 ───────────────────────────────────────────────────────
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = [
    'PingFang SC', 'STHeiti', 'Noto Sans CJK SC',
    'SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans'
]

print("=" * 70)
print("实验五：法律裁判文书中的案情要素贝叶斯分类")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════════
# DV 标签中文映射字典（婚姻家庭领域裁判文书标准要素）
# ══════════════════════════════════════════════════════════════════════════════
DV_LABEL_MAP = {
    'DV1':  '子女抚养',
    'DV2':  '抚养权归属',
    'DV3':  '财产分割',
    'DV4':  '抚养费',
    'DV5':  '房产处理',
    'DV6':  '分居事实',
    'DV7':  '起诉离婚',
    'DV8':  '探望权',
    'DV9':  '准予离婚',
    'DV10': '共同债务',
    'DV11': '婚前财产',
    'DV12': '解除婚姻',
    'DV13': '过错行为',
    'DV14': '财产隐匿',
    'DV15': '经济补偿',
    'DV16': '拒不履行',
    'DV17': '损害赔偿',
    'DV18': '遗弃行为',
    'DV19': '抚养变更',
    'DV20': '个人财产',
}

# ══════════════════════════════════════════════════════════════════════════════
# 第一步：数据加载
# ══════════════════════════════════════════════════════════════════════════════
print("\n【第一步】数据加载与探索")
print("-" * 70)

# 候选文件路径（支持 Mac / Windows / Linux）
CANDIDATE_PATHS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '离婚诉讼文本_part.json'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '离婚诉讼文本.json'),
    './离婚诉讼文本_part.json',
    './离婚诉讼文本.json',
]


def load_data_flatten(path):
    """
    加载数据：文件每行是一个JSON数组（array of dicts），
    将所有行展平成单条记录列表，并做标签映射。
    返回：(texts, labels) 两个列表
    """
    all_records = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, list):
                    all_records.extend(obj)
                elif isinstance(obj, dict):
                    all_records.append(obj)
            except json.JSONDecodeError as e:
                print(f"  ⚠️  第{i}行解析失败，已跳过：{e}")

    texts, labels = [], []
    for item in all_records:
        if not isinstance(item, dict):
            continue
        sentence = item.get('sentence', '').strip()
        raw_labels = item.get('labels', [])

        # 跳过空句子
        if not sentence:
            continue

        # 多标签取第一个；空标签归为"无标注"
        if raw_labels:
            label_code = raw_labels[0]
            label_cn = DV_LABEL_MAP.get(label_code, label_code)
        else:
            label_cn = '无标注'

        texts.append(sentence)
        labels.append(label_cn)

    return texts, labels


# 依次尝试路径
raw_texts, raw_labels = [], []
data_source = None
for p in CANDIDATE_PATHS:
    if os.path.exists(p):
        try:
            raw_texts, raw_labels = load_data_flatten(p)
            if raw_texts:
                data_source = p
                print(f"✓ 加载成功：{p}  （{len(raw_texts)} 条记录）")
                break
        except Exception as e:
            print(f"  ✗ 读取失败（{p}）：{e}")

if not raw_texts:
    raise FileNotFoundError("未找到数据文件！请将 '离婚诉讼文本_part.json' 放在脚本同目录。")

# ── 过滤"无标注"类别（可按需保留） ───────────────────────────────────────────
REMOVE_UNLABELED = True
if REMOVE_UNLABELED:
    filtered = [(t, l) for t, l in zip(raw_texts, raw_labels) if l != '无标注']
    raw_texts, raw_labels = zip(*filtered) if filtered else ([], [])
    raw_texts, raw_labels = list(raw_texts), list(raw_labels)
    print(f"✓ 过滤无标注后剩余：{len(raw_texts)} 条")

# ── 数据概览 ──────────────────────────────────────────────────────────────────
label_counts = Counter(raw_labels)
print(f"\n✓ 类别数：{len(label_counts)}")
print(f"  {'类别':<14} {'样本数':>6} {'占比':>7}")
print(f"  {'-'*30}")
total = len(raw_labels)
for cls, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
    print(f"  {cls:<14} {cnt:>6}  {cnt/total*100:>6.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# 第二步：停用词加载
# ══════════════════════════════════════════════════════════════════════════════
STOPWORD_PATHS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '停用词表.txt'),
    './停用词表.txt',
]

BASIC_STOPWORDS = {
    '的', '了', '和', '是', '在', '有', '等', '被', '也', '这', '与', '对',
    '所', '为', '由', '从', '不', '就', '而', '及', '其', '到', '以', '中',
    '之', '但', '或', '该', '此', '于', '本', '并', '可', '应', '已', '向',
}

stopwords = BASIC_STOPWORDS.copy()
for sp in STOPWORD_PATHS:
    if os.path.exists(sp):
        try:
            with open(sp, 'r', encoding='utf-8') as f:
                stopwords = set(f.read().splitlines()) | BASIC_STOPWORDS
            print(f"\n✓ 停用词表已加载：{sp}（{len(stopwords)} 词）")
        except Exception:
            pass
        break
else:
    print(f"\n⚠️  未找到停用词表，使用基础停用词（{len(stopwords)} 词）")

# ══════════════════════════════════════════════════════════════════════════════
# 第三步：文本预处理（清洗 → 分词 → 去停用词）
# ══════════════════════════════════════════════════════════════════════════════
print("\n【第二步】文本预处理（清洗 → 分词 → 去停用词）")
print("-" * 70)

jieba.setLogLevel('WARN')


def preprocess(text: str) -> str:
    """清洗文本 → jieba 分词 → 去停用词 → 空格拼接"""
    # 保留汉字与英文字母，去除标点、数字等噪声
    text = re.sub(r'[^\u4e00-\u9fffa-zA-Z]', '', str(text))
    words = [w for w in jieba.cut(text) if w not in stopwords and len(w) > 1]
    return ' '.join(words)


processed = [preprocess(t) for t in raw_texts]
print(f"✓ 预处理完成，共 {len(processed)} 条")
print(f"\n  原文示例：{raw_texts[0][:50]}...")
print(f"  处理结果：{processed[0][:70]}...")

# ══════════════════════════════════════════════════════════════════════════════
# 第四步：特征提取（BoW 与 TF-IDF）
# ══════════════════════════════════════════════════════════════════════════════
print("\n【第三步】特征提取（BoW 与 TF-IDF）")
print("-" * 70)

N_FEATURES = 300

bow_vec = CountVectorizer(max_features=N_FEATURES)
X_bow = bow_vec.fit_transform(processed)

tfidf_vec = TfidfVectorizer(max_features=N_FEATURES)
X_tfidf = tfidf_vec.fit_transform(processed)

le = LabelEncoder()
y = le.fit_transform(raw_labels)

print(f"  BoW   矩阵：{X_bow.shape}")
print(f"  TF-IDF矩阵：{X_tfidf.shape}")
print(f"  类别列表：{list(le.classes_)}")

# ══════════════════════════════════════════════════════════════════════════════
# 第五步：数据划分（自适应分层策略）
# ══════════════════════════════════════════════════════════════════════════════
print("\n【第四步】数据划分")
print("-" * 70)

min_class_count = min(Counter(y.tolist()).values())
n_classes = len(le.classes_)
n_test = max(int(len(y) * 0.2), n_classes)
use_stratify = (min_class_count >= 2) and (len(y) - n_test >= n_classes)

split_kwargs = dict(test_size=n_test, random_state=42)
if use_stratify:
    split_kwargs['stratify'] = y
    print(f"  使用分层抽样")
else:
    print(f"  使用普通划分（样本较少，最小类别数={min_class_count}）")

X_bow_tr,   X_bow_te,   y_tr, y_te = train_test_split(X_bow,   y, **split_kwargs)
X_tfidf_tr, X_tfidf_te, _,    _    = train_test_split(X_tfidf, y, **split_kwargs)

print(f"  训练集：{len(y_tr)} 条  测试集：{len(y_te)} 条")

# ══════════════════════════════════════════════════════════════════════════════
# 第六步：模型训练与评估
# ══════════════════════════════════════════════════════════════════════════════
print("\n【第五步】模型训练与评估（MultinomialNB，alpha=1.0）")
print("-" * 70)

results = {}
for name, X_tr, X_te in [
    ('MNB + BoW',    X_bow_tr,   X_bow_te),
    ('MNB + TF-IDF', X_tfidf_tr, X_tfidf_te),
]:
    t0 = time.time()
    clf = MultinomialNB(alpha=1.0)
    clf.fit(X_tr, y_tr)
    elapsed = time.time() - t0

    y_pred_tr = clf.predict(X_tr)
    y_pred_te = clf.predict(X_te)

    tr_acc = accuracy_score(y_tr, y_pred_tr)
    te_acc = accuracy_score(y_te, y_pred_te)
    prec   = precision_score(y_te, y_pred_te, average='weighted', zero_division=0)
    rec    = recall_score(y_te,  y_pred_te, average='weighted', zero_division=0)
    f1     = f1_score(y_te,     y_pred_te, average='weighted', zero_division=0)

    min_tr_cls = min(Counter(y_tr.tolist()).values())
    n_cv = min(5, min_tr_cls, len(y_tr))
    cv_scores = (cross_val_score(MultinomialNB(alpha=1.0), X_tr, y_tr,
                                 cv=n_cv, scoring='accuracy')
                 if n_cv >= 2 else np.array([te_acc]))

    results[name] = dict(
        clf=clf, X_tr=X_tr, X_te=X_te,
        y_pred_te=y_pred_te,
        tr_acc=tr_acc, te_acc=te_acc,
        prec=prec, rec=rec, f1=f1, time=elapsed,
        cv_mean=cv_scores.mean(), cv_std=cv_scores.std(), n_cv=n_cv,
    )

    print(f"\n  【{name}】")
    print(f"    训练准确率：{tr_acc:.4f}  测试准确率：{te_acc:.4f}")
    print(f"    Precision：{prec:.4f}  Recall：{rec:.4f}  F1：{f1:.4f}")
    print(f"    {n_cv}折CV：{cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
          f"  训练时间：{elapsed:.4f}s")

# 最优模型
best_name = max(results, key=lambda k: results[k]['te_acc'])
best = results[best_name]
print(f"\n  ★ 最优模型：{best_name}（测试准确率 {best['te_acc']:.4f}）")
print(f"\n  详细分类报告（{best_name}，测试集）：")
# 仅输出测试集中出现的类别
present_classes = sorted(set(y_te) | set(best['y_pred_te']))
present_names = le.inverse_transform(present_classes)
print(classification_report(y_te, best['y_pred_te'],
                             labels=present_classes,
                             target_names=present_names,
                             digits=4, zero_division=0))

# ══════════════════════════════════════════════════════════════════════════════
# 第七步：可视化（四子图）
# ══════════════════════════════════════════════════════════════════════════════
print("\n【第六步】可视化输出")
print("-" * 70)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle('实验五：朴素贝叶斯法律文本分类综合分析', fontsize=15, fontweight='bold', y=0.98)

# ── 子图1：混淆矩阵 ──────────────────────────────────────────────────────────
ax = axes[0, 0]
# 只用测试集中实际出现的类别避免维度不匹配
present_cls_idx = sorted(set(int(v) for v in y_te) | set(int(v) for v in best['y_pred_te']))
present_cls_names = list(le.inverse_transform(present_cls_idx))
cm = confusion_matrix(y_te, best['y_pred_te'], labels=present_cls_idx)
row_sums = cm.sum(axis=1, keepdims=True)
cm_norm = np.where(row_sums > 0, cm.astype(float) / row_sums, 0.0)
annot = np.array([
    [f'{cm[i,j]}\n({cm_norm[i,j]:.1%})' for j in range(len(present_cls_names))]
    for i in range(len(present_cls_names))
])
sns.heatmap(cm_norm, annot=annot, fmt='', cmap='Blues', ax=ax,
            xticklabels=present_cls_names, yticklabels=present_cls_names,
            linewidths=0.4, cbar_kws={'label': '归一化比例', 'shrink': 0.8},
            annot_kws={'size': 7})
ax.set_title(f'混淆矩阵（{best_name}）', fontsize=12, fontweight='bold')
ax.set_xlabel('预测标签', fontsize=10)
ax.set_ylabel('真实标签', fontsize=10)
ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right', fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

# ── 子图2：BoW vs TF-IDF 性能对比柱状图 ────────────────────────────────────
ax = axes[0, 1]
model_names = list(results.keys())
metrics_dict = {
    '训练准确率': [results[n]['tr_acc'] for n in model_names],
    '测试准确率': [results[n]['te_acc'] for n in model_names],
    'F1-Score':   [results[n]['f1']     for n in model_names],
}
x = np.arange(len(model_names))
w = 0.25
colors = ['#5B9BD5', '#ED7D31', '#70AD47']
for i, (metric, vals) in enumerate(metrics_dict.items()):
    bars = ax.bar(x + i * w, vals, w, label=metric, color=colors[i], alpha=0.85,
                  edgecolor='white', linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + w / 2, v + 0.005, f'{v:.3f}',
                ha='center', va='bottom', fontsize=8)
ax.set_xticks(x + w)
ax.set_xticklabels(model_names, fontsize=10)
ax.set_ylim(0, 1.15)
ax.set_ylabel('分数', fontsize=10)
ax.set_title('BoW vs TF-IDF 性能对比', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.grid(axis='y', alpha=0.3)

# ── 子图3：各类别 Precision / Recall / F1 ────────────────────────────────────
from sklearn.metrics import precision_recall_fscore_support

ax = axes[1, 0]
prec_c, rec_c, f1_c, sup_c = precision_recall_fscore_support(
    y_te, best['y_pred_te'],
    labels=np.arange(len(le.classes_)),
    average=None, zero_division=0
)
x2 = np.arange(len(le.classes_))
w2 = 0.25
ax.bar(x2 - w2, prec_c, w2, label='Precision', color='#5B9BD5', alpha=0.85)
ax.bar(x2,       rec_c,  w2, label='Recall',    color='#ED7D31', alpha=0.85)
ax.bar(x2 + w2,  f1_c,   w2, label='F1-Score',  color='#70AD47', alpha=0.85)
ax.set_xticks(x2)
ax.set_xticklabels(le.classes_, rotation=35, ha='right', fontsize=8)
ax.set_ylim(0, 1.25)
ax.set_ylabel('分数', fontsize=10)
ax.set_title('各类别 Precision / Recall / F1（测试集）', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
for i, s in enumerate(sup_c):
    ax.text(i, -0.12, f'n={s}', ha='center', fontsize=7, color='gray',
            transform=ax.get_xaxis_transform())

# ── 子图4：TF-IDF 各类 Top5 特征词权重 ──────────────────────────────────────
ax = axes[1, 1]
best_clf = best['clf']
feat_names = np.array(tfidf_vec.get_feature_names_out())
TOP_K = 5
color_map = plt.cm.tab10(np.linspace(0, 0.9, len(le.classes_)))
bar_labels, bar_vals, bar_colors = [], [], []

for cls_i, cls_name in enumerate(le.classes_):
    log_probs = best_clf.feature_log_prob_[cls_i]
    top_idx = np.argsort(log_probs)[-TOP_K:][::-1]
    for idx in top_idx:
        bar_labels.append(f'{cls_name}·{feat_names[idx]}')
        bar_vals.append(log_probs[idx])
        bar_colors.append(color_map[cls_i])

ax.barh(range(len(bar_labels)), bar_vals, color=bar_colors, alpha=0.82, edgecolor='white')
ax.set_yticks(range(len(bar_labels)))
ax.set_yticklabels(bar_labels, fontsize=7)
ax.invert_yaxis()
ax.set_xlabel('对数概率权重', fontsize=10)
ax.set_title('各类别 Top5 特征词权重（TF-IDF）', fontsize=12, fontweight='bold')

from matplotlib.patches import Patch
legend_handles = [Patch(color=color_map[i], label=cls) for i, cls in enumerate(le.classes_)]
ax.legend(handles=legend_handles, fontsize=7, loc='lower right', ncol=2)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out_png = os.path.join(OUT_DIR, 'exp5_bayes_results.png')
plt.savefig(out_png, dpi=200, bbox_inches='tight')
print(f"✓ 可视化图表已保存：{out_png}")
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 第八步：汇总输出
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("实验汇总")
print("=" * 70)
print(f"  数据来源   ：{data_source}")
print(f"  总样本数   ：{len(raw_texts)}")
print(f"  类别数     ：{len(le.classes_)}")
print(f"  最优模型   ：{best_name}")
print(f"  测试准确率 ：{best['te_acc']:.4f}")
print(f"  Precision  ：{best['prec']:.4f}")
print(f"  Recall     ：{best['rec']:.4f}")
print(f"  F1-Score   ：{best['f1']:.4f}")
print(f"  CV均值     ：{best['cv_mean']:.4f} ± {best['cv_std']:.4f}")
print(f"  过拟合差值 ：{(best['tr_acc'] - best['te_acc']):.4f}")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════════
# 附：打印各类别指标（便于填写实验报告）
# ══════════════════════════════════════════════════════════════════════════════
print("\n【各类别详细指标（可直接填入实验报告）】")
print(f"  {'类别':<14} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>8}")
print(f"  {'-' * 56}")
for cls_name, p, r, f, s in zip(le.classes_, prec_c, rec_c, f1_c, sup_c):
    print(f"  {cls_name:<14} {p:>10.4f} {r:>10.4f} {f:>10.4f} {s:>8}")
