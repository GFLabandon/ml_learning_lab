# -*- coding: utf-8 -*-
"""
实验五：法律裁判文书中的案情要素贝叶斯分类
数据：离婚诉讼文本.json（JSON Lines格式，每行一个JSON对象）
模型：TF-IDF + 多项式朴素贝叶斯
平台：macOS / Linux / Windows 跨平台
"""

import json, os, re, time, warnings
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)
import jieba

warnings.filterwarnings('ignore')

# ─── 字体：macOS优先PingFang，其余平台用SimHei/Noto ───────────────────────
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = [
    'PingFang SC', 'STHeiti', 'Noto Sans CJK SC',
    'SimHei', 'Arial Unicode MS', 'DejaVu Sans'
]

print("=" * 70)
print("实验五：法律裁判文书中的案情要素贝叶斯分类")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════
# 1. 数据加载（支持标准JSON / JSON Lines 两种格式）
# ══════════════════════════════════════════════════════════════════════════
print("\n【第一步】数据加载与探索")
print("-" * 70)

# 候选路径（会依次尝试）
CANDIDATE_PATHS = [
    os.path.join(os.path.dirname(__file__), '离婚诉讼文本.json'),
    './离婚诉讼文本.json',
    os.path.expanduser('~/ml_exp05_bayes_legal/离婚诉讼文本.json'),
]

def load_jsonlines(path):
    """读取JSON Lines格式（每行一个JSON对象）"""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ⚠️  第{i}行解析失败，已跳过：{e}")
    return records

def load_data(path):
    """
    自动识别格式：
    - 标准JSON数组  → json.load
    - JSON Lines    → 逐行读取
    - 单对象JSON    → 包装成列表
    """
    with open(path, 'r', encoding='utf-8') as f:
        raw = f.read().strip()

    # 尝试标准 JSON
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            return obj
        return [obj]
    except json.JSONDecodeError:
        pass

    # 回退到 JSON Lines
    return load_jsonlines(path)

def parse_records(records):
    """
    从记录列表中提取 (sentence, label)。
    支持字段：sentence/text/content + labels/label/category
    多标签取第一个。
    """
    texts, labels = [], []
    for item in records:
        if not isinstance(item, dict):
            continue
        text = (item.get('sentence') or item.get('text') or item.get('content', ''))
        label_raw = item.get('labels') or item.get('label') or item.get('category') or ['其他']
        label = label_raw[0] if isinstance(label_raw, list) else str(label_raw)
        if text and label:
            texts.append(str(text))
            labels.append(label)
    return texts, labels

# ── 查找文件 ──────────────────────────────────────────────────────────────
raw_texts, raw_labels = [], []
data_source = "模拟数据"

for p in CANDIDATE_PATHS:
    if os.path.exists(p):
        try:
            records = load_data(p)
            raw_texts, raw_labels = parse_records(records)
            if raw_texts:
                data_source = p
                print(f"✓ 加载成功：{p}  （{len(raw_texts)} 条）")
                break
        except Exception as e:
            print(f"  ✗ 读取失败（{p}）：{e}")

# ── 若真实数据不足，注入模拟数据做补充 ──────────────────────────────────
FALLBACK = [
    ("本案系离婚纠纷案件，原告与被告于2005年登记结婚，婚后因性格不合产生矛盾。",  "案由"),
    ("本离婚案件由本院依法受理，现已查明双方于2003年结婚，育有一子。",             "案由"),
    ("两人婚后感情逐渐破裂，多次协议离婚未果，故原告诉至本院。",                  "案由"),
    ("原告请求判决离婚，共同财产平均分割，婚生子女由原告抚养。",                  "诉讼请求"),
    ("原告诉称：判决原被告离婚，子女抚养费每月2000元，房产归原告所有。",           "诉讼请求"),
    ("原告提出：一、离婚；二、孩子抚养权归我；三、被告赔偿精神损害5万元。",        "诉讼请求"),
    ("事实与理由：被告长期不归，夫妻感情彻底破裂，已不具备共同生活基础。",        "事实与理由"),
    ("婚后被告多次家暴，经民政局调解无效，原告认为夫妻感情已完全破裂。",          "事实与理由"),
    ("双方感情不和已有数年，被告与他人同居，严重伤害了原告感情。",                "事实与理由"),
    ("经本院审理查明，原被告于2010年登记结婚，2018年开始分居，现已超过两年。",    "法院查明"),
    ("法院查明：原告所述事实属实，双方婚姻关系确已名存实亡，感情彻底破裂。",       "法院查明"),
    ("经审查，被告确存在家暴行为，有医院诊断证明及证人证言予以佐证。",             "法院查明"),
    ("本院判决：准予原告与被告离婚，婚生子女由原告抚养，共同财产各半分割。",       "判决结果"),
    ("判决如下：一、准予离婚；二、房产归被告，被告补偿原告20万元；三、子女由原告抚养。", "判决结果"),
    ("依照《民法典》第1079条，判决准予原告与被告解除婚姻关系。",                  "判决结果"),
]

if len(raw_texts) < 20:
    print(f"  ⚠️  真实数据不足（{len(raw_texts)} 条），自动补充模拟样本")
    for t, l in FALLBACK:
        raw_texts.append(t)
        raw_labels.append(l)

print(f"\n✓ 数据概览")
print(f"  总样本数：{len(raw_texts)}")
label_counts = Counter(raw_labels)
print(f"  类别数：{len(label_counts)}")
for cls, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
    print(f"    {cls}: {cnt} 条")

# ══════════════════════════════════════════════════════════════════════════
# 2. 停用词加载
# ══════════════════════════════════════════════════════════════════════════
STOPWORD_PATHS = [
    os.path.join(os.path.dirname(__file__), '停用词表.txt'),
    './停用词表.txt',
]

stopwords = {'的', '了', '和', '是', '在', '有', '等', '被', '也',
             '这', '与', '对', '所', '为', '由', '从', '不', '就', '而'}

for sp in STOPWORD_PATHS:
    if os.path.exists(sp):
        try:
            with open(sp, 'r', encoding='utf-8') as f:
                stopwords = set(f.read().splitlines())
            print(f"✓ 停用词表已加载：{sp}（{len(stopwords)} 词）")
        except Exception:
            pass
        break

# ══════════════════════════════════════════════════════════════════════════
# 3. 文本预处理
# ══════════════════════════════════════════════════════════════════════════
print("\n【第二步】文本预处理（清洗 → 分词 → 去停用词）")
print("-" * 70)

def preprocess(text):
    text = re.sub(r'[^\u4e00-\u9fffa-zA-Z0-9]', '', str(text))
    words = [w for w in jieba.cut(text) if w not in stopwords and len(w) > 1]
    return ' '.join(words)

jieba.setLogLevel('WARN')
processed = [preprocess(t) for t in raw_texts]
print(f"✓ 预处理完成，示例：")
print(f"  原文：{raw_texts[0][:60]}...")
print(f"  处理：{processed[0][:80]}...")

# ══════════════════════════════════════════════════════════════════════════
# 4. 特征提取
# ══════════════════════════════════════════════════════════════════════════
print("\n【第三步】特征提取（BoW 与 TF-IDF）")
print("-" * 70)

N_FEATURES = 200

bow_vec = CountVectorizer(max_features=N_FEATURES)
X_bow = bow_vec.fit_transform(processed)

tfidf_vec = TfidfVectorizer(max_features=N_FEATURES)
X_tfidf = tfidf_vec.fit_transform(processed)

le = LabelEncoder()
y = le.fit_transform(raw_labels)

print(f"  BoW  矩阵：{X_bow.shape}")
print(f"  TF-IDF矩阵：{X_tfidf.shape}")
print(f"  类别编码：{dict(zip(le.classes_, le.transform(le.classes_)))}")

# ══════════════════════════════════════════════════════════════════════════
# 5. 数据划分（自适应：样本足够时分层，否则普通划分）
# ══════════════════════════════════════════════════════════════════════════
print("\n【第四步】数据划分")
print("-" * 70)

min_class_count = min(Counter(y.tolist()).values())
n_classes = len(le.classes_)

# 分层要求：测试集样本数 >= 类别数，且每类至少2条
n_test = max(int(len(y) * 0.2), n_classes)
use_stratify = (min_class_count >= 2) and (len(y) - n_test >= n_classes)

split_kwargs = dict(test_size=n_test, random_state=42)
if use_stratify:
    split_kwargs['stratify'] = y

X_bow_tr,   X_bow_te,   y_tr, y_te = train_test_split(X_bow,   y, **split_kwargs)
X_tfidf_tr, X_tfidf_te, _,    _    = train_test_split(X_tfidf, y, **split_kwargs)

print(f"  训练集：{len(y_tr)}  测试集：{len(y_te)}"
      f"  {'（分层抽样）' if use_stratify else '（普通划分，样本较少）'}")

# ══════════════════════════════════════════════════════════════════════════
# 6. 模型训练与评估
# ══════════════════════════════════════════════════════════════════════════
print("\n【第五步】模型训练与评估")
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
    prec = precision_score(y_te, y_pred_te, average='weighted', zero_division=0)
    rec  = recall_score(y_te,  y_pred_te, average='weighted', zero_division=0)
    f1   = f1_score(y_te,    y_pred_te, average='weighted', zero_division=0)

    # 交叉验证（cv折数 = min(5, 每类最少样本数, 训练集大小)）
    min_tr_cls = min(Counter(y_tr.tolist()).values())
    n_cv = min(5, min_tr_cls, len(y_tr))
    cv = (cross_val_score(MultinomialNB(alpha=1.0), X_tr, y_tr,
                          cv=n_cv, scoring='accuracy')
          if n_cv >= 2 else np.array([te_acc]))

    results[name] = dict(
        clf=clf, X_tr=X_tr, X_te=X_te,
        y_pred_te=y_pred_te, tr_acc=tr_acc, te_acc=te_acc,
        prec=prec, rec=rec, f1=f1, time=elapsed,
        cv_mean=cv.mean(), cv_std=cv.std()
    )
    print(f"\n  【{name}】")
    print(f"    训练准确率：{tr_acc:.4f}  测试准确率：{te_acc:.4f}")
    print(f"    Precision：{prec:.4f}  Recall：{rec:.4f}  F1：{f1:.4f}")
    print(f"    {n_cv}折CV：{cv.mean():.4f} ± {cv.std():.4f}  训练时间：{elapsed:.4f}s")

# 最优模型
best_name = max(results, key=lambda k: results[k]['te_acc'])
best = results[best_name]
print(f"\n  ★ 最优模型：{best_name}（测试准确率 {best['te_acc']:.4f}）")

# 详细分类报告
print(f"\n  详细分类报告（{best_name}，测试集）：")
print(classification_report(y_te, best['y_pred_te'],
                             target_names=le.classes_, digits=4, zero_division=0))

# ══════════════════════════════════════════════════════════════════════════
# 7. 可视化（保存为 PNG）
# ══════════════════════════════════════════════════════════════════════════
print("\n【第六步】可视化输出")
print("-" * 70)

OUT_DIR = os.path.dirname(__file__)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('实验五：朴素贝叶斯法律文本分类分析', fontsize=15, fontweight='bold')

# ── 子图1：混淆矩阵 ──────────────────────────────────────────────────────
ax = axes[0, 0]
cm = confusion_matrix(y_te, best['y_pred_te'])
cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax,
            xticklabels=le.classes_, yticklabels=le.classes_,
            linewidths=0.5)
ax.set_title(f'混淆矩阵（{best_name}）', fontsize=12, fontweight='bold')
ax.set_xlabel('预测标签'); ax.set_ylabel('真实标签')
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

# ── 子图2：BoW vs TF-IDF 性能对比 ────────────────────────────────────────
ax = axes[0, 1]
names = list(results.keys())
metrics_vals = {
    '训练准确率': [results[n]['tr_acc'] for n in names],
    '测试准确率': [results[n]['te_acc'] for n in names],
    'F1-Score':   [results[n]['f1']     for n in names],
}
x = np.arange(len(names)); w = 0.25
colors = ['#5B9BD5', '#ED7D31', '#70AD47']
for i, (metric, vals) in enumerate(metrics_vals.items()):
    bars = ax.bar(x + i*w, vals, w, label=metric, color=colors[i], alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + w/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
ax.set_xticks(x + w); ax.set_xticklabels(names, fontsize=10)
ax.set_ylim(0, 1.1); ax.set_ylabel('分数')
ax.set_title('BoW vs TF-IDF 性能对比', fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

# ── 子图3：各类别 F1 ─────────────────────────────────────────────────────
ax = axes[1, 0]
from sklearn.metrics import precision_recall_fscore_support
prec_c, rec_c, f1_c, sup_c = precision_recall_fscore_support(
    y_te, best['y_pred_te'], labels=np.arange(len(le.classes_)),
    average=None, zero_division=0)
x2 = np.arange(len(le.classes_)); w2 = 0.25
ax.bar(x2 - w2, prec_c, w2, label='Precision', color='#5B9BD5', alpha=0.85)
ax.bar(x2,      rec_c,  w2, label='Recall',    color='#ED7D31', alpha=0.85)
ax.bar(x2 + w2, f1_c,   w2, label='F1',        color='#70AD47', alpha=0.85)
ax.set_xticks(x2); ax.set_xticklabels(le.classes_, rotation=20, ha='right', fontsize=9)
ax.set_ylim(0, 1.2); ax.set_ylabel('分数')
ax.set_title('各类别性能指标', fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
for i, (p, r, f, s) in enumerate(zip(prec_c, rec_c, f1_c, sup_c)):
    ax.text(i, -0.1, f'n={s}', ha='center', fontsize=8, color='gray',
            transform=ax.get_xaxis_transform())

# ── 子图4：TF-IDF 各类 Top词权重 ─────────────────────────────────────────
ax = axes[1, 1]
best_clf = best['clf']
feat_names = np.array(tfidf_vec.get_feature_names_out())
top_k = 5
colors_cls = plt.cm.tab10(np.linspace(0, 0.8, len(le.classes_)))
bar_labels, bar_vals, bar_colors = [], [], []

for cls_i, cls_name in enumerate(le.classes_):
    w_arr = best_clf.feature_log_prob_[cls_i]
    top_idx = np.argsort(w_arr)[-top_k:][::-1]
    for idx in top_idx:
        bar_labels.append(f'{cls_name}·{feat_names[idx]}')
        bar_vals.append(w_arr[idx])
        bar_colors.append(colors_cls[cls_i])

ax.barh(range(len(bar_labels)), bar_vals, color=bar_colors, alpha=0.8)
ax.set_yticks(range(len(bar_labels)))
ax.set_yticklabels(bar_labels, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('对数概率权重')
ax.set_title('各类别 Top5 特征词权重', fontsize=12, fontweight='bold')

# 图例（类别颜色）
from matplotlib.patches import Patch
legend_handles = [Patch(color=colors_cls[i], label=cls)
                  for i, cls in enumerate(le.classes_)]
ax.legend(handles=legend_handles, fontsize=8, loc='lower right')

plt.tight_layout()
out_png = os.path.join(OUT_DIR, 'exp5_legal_bayes_results.png')
plt.savefig(out_png, dpi=200, bbox_inches='tight')
print(f"✓ 图表已保存：{out_png}")
plt.show()

# ══════════════════════════════════════════════════════════════════════════
# 8. 汇总
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("实验总结")
print("=" * 70)
print(f"  数据来源   ：{data_source}")
print(f"  总样本数   ：{len(raw_texts)}")
print(f"  最优模型   ：{best_name}")
print(f"  测试准确率 ：{best['te_acc']:.4f}")
print(f"  Precision  ：{best['prec']:.4f}")
print(f"  Recall     ：{best['rec']:.4f}")
print(f"  F1-Score   ：{best['f1']:.4f}")
print(f"  CV均值     ：{best['cv_mean']:.4f} ± {best['cv_std']:.4f}")
print(f"  过拟合差值 ：{(best['tr_acc']-best['te_acc']):.4f}")
print("=" * 70)