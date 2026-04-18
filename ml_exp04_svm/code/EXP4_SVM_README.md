# 🎓 实验四：基于SVM的手写数字识别

## 📋 实验概述

本实验通过MNIST手写数字数据集，系统地研究支持向量机(SVM)算法，并对比不同核函数和多分类策略的性能差异。

## 🎯 实验目标

1. 理解SVM的基本原理（最大间隔分类器）
2. 掌握4种常见核函数的特点和适用场景
3. 对比OVO和OVR两种多分类策略的优缺点
4. 实现0-9手写数字的10分类预测
5. 分析模型性能并进行参数优化建议

## 📁 文件说明

### 核心代码文件

**exp4_svm_handwriting_digits.py** (完整实验代码)
```
主要功能：
✓ 加载MNIST/digits数据集
✓ 数据预处理（归一化、标准化）
✓ 训练8种SVM模型组合（4核函数 × 2策略）
✓ 统计训练时间、准确率等性能指标
✓ 生成对比图表（准确率、训练时间、混淆矩阵）
✓ 详细的结果分析与建议

核函数：linear, poly, rbf, sigmoid
策略：OVR (One-vs-Rest), OVO (One-vs-One)
```

## 🚀 快速开始

### 环境安装

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### 运行代码

```bash
python exp4_svm_handwriting_digits.py
```

代码将自动：
1. 查找并加载MNIST数据集（如未找到，使用scikit-learn的digits数据集）
2. 训练8种SVM模型组合
3. 输出性能对比表格
4. 生成结果可视化图表（svm_handwriting_digits_analysis.png）
5. 进行详细的分析与建议

## 📊 实验设计

### 对比维度

#### 核函数对比
| 核函数 | 特点 | 适用场景 | 复杂度 |
|--------|------|---------|--------|
| **Linear** | 线性映射，简单快速 | 线性可分数据 | 低 |
| **Poly** | 多项式映射，灵活 | 中等非线性问题 | 中 |
| **RBF** | 径向基函数，最灵活 | 高度非线性问题 | 高 |
| **Sigmoid** | 类神经网络 | 特殊场景 | 高 |

#### 多分类策略对比
| 策略 | 分类器数量 | 数据特点 | 训练速度 | 准确率 |
|------|-----------|---------|---------|--------|
| **OVR** | k个 | 各类不平衡 | 快 | 中等 |
| **OVO** | k(k-1)/2个 | 各类平衡 | 慢 | 通常更高 |

对于10分类（手写数字0-9）：
- OVR：10个分类器
- OVO：45个分类器（10×9/2）

### 数据处理流程

```
原始数据 → 归一化 → 标准化 → 划分 → 训练/验证/测试
↓
像素值[0,255] → [0,1] → μ=0,σ=1 → 80%/10%/10%
```

## 📈 关键概念

### SVM基本原理

支持向量机通过以下目标函数实现二分类：

```
最小化：||w||²/2 + C Σ ξᵢ

约束条件：yᵢ(wᵀφ(xᵢ) + b) ≥ 1 - ξᵢ
```

其中：
- w：超平面的法向量
- b：偏置项
- C：正则化参数（控制容忍度）
- ξ：松弛变量（允许少量错误分类）
- φ()：核函数

### 核函数原理

核函数通过隐式映射，将低维非线性问题转化为高维线性问题：

```
K(xᵢ, xⱼ) = φ(xᵢ)ᵀφ(xⱼ)
```

不需要显式计算映射函数φ()，直接计算内积。

### OVO vs OVR

**OVR（一对多）**：
- 类别i vs 其余类别
- 分类器总数：k
- 预测：选择得分最高的分类器

**OVO（一对一）**：
- 每对类别(i,j)一个分类器
- 分类器总数：k(k-1)/2
- 预测：投票机制（哪个类别被选中次数最多）

## 🔍 代码使用指南

### 基本使用

```python
# 导入必要的库
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# 加载数据
data = load_digits()  # 或 load_mnist()
X, y = data.data, data.target

# 预处理
X = (X - X.min()) / (X.max() - X.min())  # 归一化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 训练SVM
svm = SVC(kernel='rbf', decision_function_shape='ovr', C=1.0)
svm.fit(X_train, y_train)

# 预测和评估
accuracy = svm.score(X_test, y_test)
```

### 使用自己的MNIST数据

如果你有mnist.pkl.gz文件：

```python
import pickle
import gzip

# 加载数据
with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_data, valid_data, test_data = pickle.load(f, encoding='bytes')

X_train, y_train = train_data[0], train_data[1]
X_test, y_test = test_data[0], test_data[1]
```

## 📊 输出说明

### 控制台输出

1. **数据信息**：样本数、特征数、类别分布
2. **模型性能表**：逐个模型的训练时间、三个准确率
3. **最优模型**：核函数、策略、各项指标
4. **分类报告**：每个数字的精确率、召回率、F1分数
5. **总体指标**：加权平均的precision, recall, F1

### 可视化输出（4个子图）

1. **核函数准确率对比**：柱状图，OVR vs OVO
2. **核函数训练时间对比**：柱状图，OVR vs OVO
3. **所有模型组合对比**：柱状图，8个模型的测试准确率
4. **混淆矩阵**：热力图，显示每个数字的分类情况

## 🎯 预期结果

### 典型性能（使用scikit-learn digits数据集）

- **Linear + OVR**：准确率 ~95%，训练时间 ~0.01s
- **RBF + OVR**：准确率 ~97%，训练时间 ~0.1s
- **RBF + OVO**：准确率 ~98%，训练时间 ~0.5s

### MNIST数据集（如可用）

- **Linear + OVR**：准确率 ~92%，训练时间 ~2s
- **RBF + OVR**：准确率 ~95%，训练时间 ~30s
- **RBF + OVO**：准确率 ~96%，训练时间 ~200s

## 💡 进阶应用

### 参数调优

```python
# 调整C参数（正则化强度）
svm = SVC(kernel='rbf', C=0.1)  # 更宽松的边界
svm = SVC(kernel='rbf', C=10.0) # 更严格的边界

# 调整gamma参数（RBF核影响范围）
svm = SVC(kernel='rbf', gamma=0.001)  # 影响范围小
svm = SVC(kernel='rbf', gamma=0.1)    # 影响范围大
```

### 网格搜索超参数

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.001, 0.1]
}
grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
```

### 特征工程

```python
# PCA降维
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# 结合PCA和SVM
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
```

## 📚 理论参考

### SVM经典书籍
- 《Support Vector Machines Succinctly》- Alex Smola
- 《Learning with Kernels》- Schölkopf & Smola

### 核心论文
- Cortes, C., & Vapnik, V. (1995). "Support-vector networks"
- Platt, J. (1999). "Fast Training of Support Vector Machines using Sequential Minimal Optimization"

### 在线资源
- [scikit-learn SVM文档](https://scikit-learn.org/stable/modules/svm.html)
- [LibSVM教程](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)

## ⚠️ 常见问题

**Q: 代码运行很慢怎么办？**
A: 这是正常的。SVM在大数据集上训练较慢，特别是OVO策略。可以：
- 减少训练样本数
- 使用Linear核（最快）
- 使用OVR策略而不是OVO
- 增加C参数（会加快训练但可能降低精度）

**Q: 如何处理内存不足的问题？**
A: 
- 减少样本数（随机抽样）
- 使用PCA降维
- 考虑使用SGDClassifier替代SVM

**Q: 如何加载标准的MNIST数据集？**
A: 代码会自动查找mnist.pkl.gz，如未找到会使用scikit-learn的digits数据集。如需标准MNIST，可在官网下载：http://yann.lecun.com/exdb/mnist/

## 📝 实验报告提示

完成代码运行后，写实验报告时可包括：

1. **实验目的**：理解SVM原理，对比核函数和多分类策略
2. **数据预处理**：描述归一化、标准化、数据划分过程
3. **模型对比**：基于输出表格分析各模型性能
4. **结果分析**：
   - 哪个核函数表现最优？为什么？
   - OVO和OVR的性能与速度差异
   - 最优模型的混淆矩阵含义
5. **结论与建议**：总结发现，提出改进方向

---

**更新日期**：2025年4月18日
**代码状态**：已测试通过，可直接运行 ✅
