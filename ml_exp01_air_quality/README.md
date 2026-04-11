# 机器学习实践：空气质量预测实验

## 📋 项目概述

本项目包含基于北京市空气质量监测数据的三个机器学习实验：
1. **一元线性回归**：基于CO浓度预测PM2.5浓度
2. **多元线性回归**：基于多个污染物浓度预测PM2.5浓度
3. **逻辑回归**：预测空气质量污染分类

## 📁 文件结构

```
├── 机器学习实践-空气质量预测实验报告.docx    # 完整的实验报告（Word格式）
├── simple_linear_regression.py               # 一元线性回归代码
├── multiple_linear_regression.py             # 多元线性回归代码
├── logistic_regression.py                    # 逻辑回归代码
├── 北京市空气质量数据.xlsx                   # 实验数据源
├── simple_linear_regression_results.png      # 一元回归结果图表
├── multiple_linear_regression_results.png    # 多元回归结果图表
├── logistic_regression_results.png           # 逻辑回归结果图表
└── README.md                                  # 本文件
```

## 🔧 环境要求

### Python版本
- Python 3.8 及以上

### 必需库
```bash
pip install numpy pandas matplotlib seaborn scikit-learn openpyxl
```

或使用requirements.txt（如果提供）：
```bash
pip install -r requirements.txt
```

## 🚀 快速开始

### 1. 一元线性回归实验

```bash
python simple_linear_regression.py
```

**功能**：
- 使用CO浓度预测PM2.5浓度
- 输出模型公式和性能指标（MSE, RMSE, R²）
- 生成4张对比图表

**输出文件**：
- `simple_linear_regression_results.png` - 包含4个子图的结果可视化

### 2. 多元线性回归实验

```bash
python multiple_linear_regression.py
```

**功能**：
- 使用CO、SO2、NO2、O3、PM10等多个特征预测PM2.5
- 分析各特征对PM2.5的影响程度
- 输出详细的性能指标和相关性分析

**输出文件**：
- `multiple_linear_regression_results.png` - 包含9个子图的结果可视化

### 3. 逻辑回归实验

```bash
python logistic_regression.py
```

**功能**：
- 预测空气质量是否污染（二分类问题）
- 目标标签：优/良 → 无污染(0)，其他 → 有污染(1)
- 使用PM2.5和PM10作为特征
- 生成决策边界、混淆矩阵、ROC曲线等

**输出文件**：
- `logistic_regression_results.png` - 包含6个子图的结果可视化

## 📊 数据说明

### 数据源
- **文件**：`北京市空气质量数据.xlsx`
- **时间范围**：2014年1月 - 2019年11月
- **样本数量**：2155条记录
- **特征列**：日期、AQI、质量等级、PM2.5、PM10、SO2、CO、NO2、O3

### 数据字段

| 字段 | 类型 | 说明 |
|------|------|------|
| 日期 | datetime | 监测日期 |
| AQI | int | 空气质量指数 |
| 质量等级 | string | 优、良、轻度污染、中度污染等 |
| PM2.5 | int | 细颗粒物浓度 (μg/m³) |
| PM10 | int | 粗颗粒物浓度 (μg/m³) |
| SO2 | int | 二氧化硫浓度 (ppb) |
| CO | float | 一氧化碳浓度 (mg/m³) |
| NO2 | int | 二氧化氮浓度 (ppb) |
| O3 | int | 臭氧浓度 (ppb) |

## 📈 主要结果总结

### 实验一：一元线性回归
- **模型方程**：PM2.5 = 58.76 × CO + 2.73
- **测试集R²**：0.6922
- **测试集RMSE**：33.43 μg/m³
- **结论**：CO与PM2.5呈正相关，但单个特征解释力有限

### 实验二：多元线性回归
- **模型方程**：PM2.5 = 32.08×CO - 0.11×SO2 + 0.45×NO2 + 0.10×O3 + 0.36×PM10 - 31.09
- **测试集R²**：0.8189
- **测试集RMSE**：25.64 μg/m³
- **主要特征**：PM10 (相关系数0.847) > CO (0.844) > NO2 (0.782)
- **结论**：多元模型性能显著提升，PM10是最强预测因子

### 实验三：逻辑回归
- **准确率**：83.53%
- **AUC分数**：0.9101
- **灵敏度**：79.03%（正确识别污染天数）
- **特异性**：86.94%（正确识别非污染天数）
- **F1-Score**：0.8055
- **结论**：模型性能很好，可用于污染预警

## 🎯 使用建议

### 在PyCharm中运行

1. **打开项目**
   - 打开PyCharm → Open → 选择项目文件夹

2. **配置Python解释器**
   - PyCharm → Preferences → Project → Python Interpreter
   - 选择Python 3.8+ 版本

3. **运行脚本**
   - 右键点击Python文件 → Run 'filename'
   - 或使用快捷键 Shift+F10 (Windows/Linux) 或 Ctrl+R (Mac)

4. **查看输出**
   - 控制台输出详细的分析结果
   - 生成PNG图表文件在同一目录

### 修改数据路径

如果数据文件不在同一目录，修改代码中的路径：

```python
# 方法1：使用绝对路径
data_path = '/path/to/北京市空气质量数据.xlsx'

# 方法2：使用相对路径
data_path = os.path.join('..', 'data', '北京市空气质量数据.xlsx')
```

### 自定义参数

你可以修改以下参数进行实验：

```python
# 一元线性回归：修改特征列
X = data[['SO2']].values  # 改用SO2预测PM2.5

# 多元线性回归：修改特征组合
features = ['CO', 'SO2', 'PM10']  # 选择部分特征

# 逻辑回归：修改分类标准
data['pollution'] = (data['AQI'] > 100).astype(int)  # 基于AQI分类
```

## 🔬 技术细节

### 一元线性回归
- **算法**：最小二乘法 (OLS)
- **类库**：scikit-learn LinearRegression
- **训练集比例**：80%
- **评估指标**：MSE, RMSE, R²

### 多元线性回归
- **算法**：最小二乘法
- **特征数**：5个
- **特征相关性分析**：Pearson相关系数
- **评估指标**：MSE, RMSE, MAE, R²

### 逻辑回归
- **算法**：基于梯度的优化
- **特征标准化**：StandardScaler
- **类权重**：balanced（处理类不平衡）
- **评估指标**：准确率、精确率、召回率、F1-Score、AUC
- **可视化**：决策边界、混淆矩阵、ROC曲线

## 📚 参考资源

### scikit-learn 文档
- [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

### Matplotlib 中文绘图
- 使用`SimHei`字体进行中文显示
- 通过`plt.rcParams`配置全局字体

## ⚠️ 常见问题

### Q: 为什么图表显示中文乱码？
A: 代码已包含中文字体配置：
```python
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
```
如果仍显示乱码，确保系统已安装SimHei字体，或改用其他支持的字体。

### Q: 如何修改训练/测试集比例？
A: 修改 `train_test_split` 的 `test_size` 参数：
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42  # 改为0.3表示70-30分割
)
```

### Q: 可以使用自己的数据吗？
A: 可以。确保数据格式与示例相同（Excel或CSV），并修改列名和路径即可。

## 📄 许可证

本项目为教学用途，基于开源库scikit-learn、pandas、matplotlib等。

## 👥 作者信息

- **课程**：机器学习实践
- **学校**：中北大学软件学院
- **指导教师**：程晓鼎

---

**更新日期**：2025年4月

如有问题或建议，欢迎反馈！
