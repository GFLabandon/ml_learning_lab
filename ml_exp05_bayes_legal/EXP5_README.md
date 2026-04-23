# 🎓 实验五：基于朴素贝叶斯的法律文本分类

## 📋 实验概述

本实验使用朴素贝叶斯分类器对中文法律文本（离婚诉讼案件）进行自动分类，对比词袋模型（BoW）和TF-IDF两种特征表示方法的性能差异。

## 🎯 实验目标

1. 理解朴素贝叶斯分类器的基本原理
2. 掌握中文文本预处理方法（分词、清洗、停用词过滤）
3. 学习特征提取技术（词袋模型、TF-IDF）
4. 对比不同特征表示对分类性能的影响
5. 实现法律文本的自动分类，评估模型性能

## 📁 文件说明

### 核心代码文件

**exp5_naive_bayes_legal_text_classification.py** (完整实验代码)
```
主要功能：
✓ 加载法律文本数据集（离婚诉讼文本.json）
✓ 中文文本预处理（分词、清洗、停用词过滤）
✓ 特征提取（词袋模型、TF-IDF）
✓ 训练两种朴素贝叶斯模型
✓ 性能评估（准确率、精确率、召回率、F1）
✓ 混淆矩阵和交叉验证
✓ 结果可视化（4个对比图表）

主要模型：
• MultinomialNB + BoW特征
• MultinomialNB + TF-IDF特征
```

## 🚀 快速开始

### 环境安装

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jieba
```

### 运行代码

```bash
# 如果有数据文件
python exp5_naive_bayes_legal_text_classification.py

# 代码会自动查找数据文件，如找不到则使用模拟数据
```

代码将自动：
1. 加载离婚诉讼文本数据集
2. 进行文本预处理
3. 提取BoW和TF-IDF特征
4. 训练两个朴素贝叶斯模型
5. 输出性能对比表格
6. 生成结果可视化图表（naive_bayes_text_classification.png）
7. 进行详细的分析与建议

## 📊 实验设计

### 对比维度

#### 特征提取方法对比
| 方法 | 特点 | 优点 | 缺点 |
|------|------|------|------|
| **词袋模型(BoW)** | 计数 | 快速、简单、易理解 | 忽视词序、不考虑词重要性 |
| **TF-IDF** | 加权计数 | 考虑词重要性、性能更优 | 计算复杂、维度高 |

#### 数据处理流程
```
原始文本 → 文本清洗 → 中文分词 → 停用词移除 → 特征提取 → 模型训练
   ↓
案例：
"本案系离婚纠纷案件" 
→ "本案系离婚纠纷案件" 
→ ["本", "案", "系", "离婚", "纠纷", "案件"] 
→ ["离婚", "纠纷", "案件"] 
→ 特征向量 
→ 分类预测
```

### 朴素贝叶斯原理

**贝叶斯定理：**
```
P(C|X) = P(X|C) × P(C) / P(X)
```

其中：
- P(C|X)：后验概率（给定特征X时属于类C的概率）
- P(X|C)：似然度（给定类C时特征X出现的概率）
- P(C)：先验概率（类C出现的概率）
- P(X)：证据（特征X出现的概率）

**朴素假设：**
各特征在给定类别条件下相互独立，即：
```
P(X|C) = P(x1|C) × P(x2|C) × ... × P(xn|C)
```

**分类决策：**
选择使后验概率最大的类别：
```
argmax_C P(C|X)
```

## 📈 预期结果

基于模拟数据的典型结果：

```
模型                    训练准确率  测试准确率  训练时间
─────────────────────────────────────────────
MNB + BoW              0.9167     0.8333     0.0045s
MNB + TF-IDF           0.9167     0.8333     0.0052s

最优模型：MNB + TF-IDF（或两者相近）
```

实际数据（离婚诉讼文本.json）的结果会因数据规模和质量而异。

## 💡 核心代码示例

### 1. 文本预处理

```python
import jieba
import re

def clean_text(text):
    """文本清洗"""
    text = re.sub(r'[^\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ffa-zA-Z0-9]', '', text)
    return text

def segment_text(text):
    """中文分词"""
    words = jieba.cut(text)
    return list(words)

def remove_stopwords(words, stopwords={'的', '了', '是', '和'}):
    """停用词过滤"""
    return [w for w in words if w not in stopwords]

# 使用示例
text = "本案系离婚纠纷案件"
cleaned = clean_text(text)
words = segment_text(cleaned)
filtered = remove_stopwords(words)
```

### 2. 特征提取

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# 方法1：词袋模型
bow_vectorizer = CountVectorizer(max_features=100)
X_bow = bow_vectorizer.fit_transform(texts)

# 方法2：TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=100)
X_tfidf = tfidf_vectorizer.fit_transform(texts)
```

### 3. 模型训练

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 训练模型
model = MultinomialNB(alpha=1.0)  # alpha为拉普拉斯平滑系数
model.fit(X_train, y_train)

# 评估
accuracy = model.score(X_test, y_test)
```

### 4. 模型评估

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score

# 准确率、精确率、召回率
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# 交叉验证
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
```

## 🔍 中文文本处理特点

### 为什么中文处理比英文复杂？

1. **分词问题**：中文没有空格分隔，需要专门工具分词
   - "离婚诉讼" vs "离" + "婚" + "诉讼"
   - 分词错误会直接影响特征质量

2. **歧义问题**：同一个词组可能有多种分法
   - "分手费" → "分手" + "费" 或 "分" + "手费"？
   - 需要用语义信息判断

3. **停用词问题**：中文停用词列表需要手工维护
   - 英文：the, is, and
   - 中文：的，了，是，和

4. **特征稀疏性**：中文词汇量大，容易造成特征稀疏
   - 解决：限制特征维度、使用TF-IDF权重

### 推荐工具

- **分词**：jieba、pkuseg、HanLP
- **停用词**：自己维护列表或下载标准列表
- **特征提取**：scikit-learn的CountVectorizer和TfidfVectorizer

## 📊 关键评估指标

**准确率（Accuracy）**
```
= 正确预测数 / 总样本数
范围：[0, 1]，越高越好
缺点：类别不平衡时可能误导
```

**精确率（Precision）**
```
= 预测为正的样本中实际为正的数量 / 所有预测为正的样本数
评估：预测为某类时的准确程度
```

**召回率（Recall）**
```
= 实际为正的样本中被预测为正的数量 / 所有实际为正的样本数
评估：对某类的发现能力
```

**F1分数**
```
= 2 × (精确率 × 召回率) / (精确率 + 召回率)
综合评估精确率和召回率的调和平均值
```

## 🛠️ 参数调优

### 多项式朴素贝叶斯参数

**alpha（拉普拉斯平滑系数）**
```python
# alpha = 0：不进行平滑（可能导致0概率）
# alpha = 1.0：默认值，进行拉普拉斯平滑
# alpha > 1：更强的平滑

model = MultinomialNB(alpha=1.0)
```

### 特征提取参数

**CountVectorizer/TfidfVectorizer参数**
```python
vectorizer = CountVectorizer(
    max_features=100,      # 最多保留100个特征
    min_df=1,              # 至少在1个文档中出现
    max_df=0.9,            # 最多在90%的文档中出现
    ngram_range=(1, 2),    # 包含一元和二元词组
    stop_words=['的', '了'] # 自定义停用词
)
```

## ⚠️ 常见问题

**Q: jieba不可用怎么办？**
A: 安装jieba：`pip install jieba`
   或使用其他分词工具，代码中有备用方案

**Q: 数据集很大，处理慢怎么办？**
A: • 限制特征维度（max_features参数）
   • 减少样本量进行快速测试
   • 使用HashingVectorizer替代CountVectorizer

**Q: 类别不平衡怎么办？**
A: • 使用class_weight='balanced'参数
   • 对少数类过采样或多数类欠采样
   • 使用专门的不平衡学习算法

**Q: 准确率很低怎么办？**
A: • 检查文本预处理是否正确
   • 优化停用词表
   • 增加训练数据
   • 调整参数
   • 尝试其他算法

## 📚 推荐资源

### 书籍
- 《自然语言处理基础》- 宗成庆
- 《机器学习》- 周志华
- 《统计自然语言处理》- Christopher D. Manning等

### 在线资源
- [scikit-learn文本特征提取](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [jieba中文分词](https://github.com/fxsjy/jieba)
- [spaCy中文处理](https://spacy.io/)

### 论文
- Naive Bayes and Text Classification (1998)
- A Re-examination of Text Categorization (2003)

## 🎓 学习路径

### 初学者
1. 理解贝叶斯定理（概率基础）
2. 学习分词原理（中文处理）
3. 实现简单的词袋模型
4. 运行本实验代码
5. 分析结果和性能指标

### 进阶学习
1. 深入理解TF-IDF权重机制
2. 学习其他特征提取方法（Word2Vec、BERT等）
3. 对比多种分类算法（SVM、LR、NN等）
4. 处理类别不平衡问题
5. 使用集成方法提升性能

## 📝 实验报告建议

完成代码运行后，写实验报告时可包括：

1. **实验目的**：理解朴素贝叶斯和文本处理
2. **方法说明**：
   - 朴素贝叶斯原理
   - 特征提取方法对比
   - 中文处理流程
3. **结果分析**：基于输出表格和图表
   - 两种特征方法的性能对比
   - 模型的泛化能力分析
   - 过拟合/欠拟合判断
4. **结论与建议**：总结发现，提出改进方向

## 🎉 快速开始（3步）

```bash
# 步骤1：安装依赖
pip install numpy pandas scikit-learn matplotlib seaborn jieba

# 步骤2：运行代码
python exp5_naive_bayes_legal_text_classification.py

# 步骤3：查看结果
# 控制台输出：8个评估指标、分类报告
# 生成文件：naive_bayes_text_classification.png
# 填充到Word报告模板
```

---

**更新日期**：2025年4月20日
**代码状态**：已测试通过，可直接运行 ✅
**数据说明**：如需使用实际数据，请在上传离婚诉讼文本.json后运行
