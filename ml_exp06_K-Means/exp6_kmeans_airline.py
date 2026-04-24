# -*- coding: utf-8 -*-
"""
实验六：航空公司客户价值的 K-Means 分析
数据：air_data.csv（航空公司客户数据）
模型：LRFMC + K-Means 聚类
平台：macOS / Linux / Windows 跨平台
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')

# ─── 字体：跨平台中文支持 ────────────────────────────────────────────────
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = [
    'PingFang SC', 'STHeiti', 'Noto Sans CJK SC',
    'SimHei', 'Arial Unicode MS', 'DejaVu Sans'
]

print("=" * 70)
print("实验六：航空公司客户价值的 K-Means 分析")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════
# 1. 数据加载
# ══════════════════════════════════════════════════════════════════════════
print("\n【第一步】数据加载与探索")
print("-" * 70)

CANDIDATE_PATHS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'air_data.csv'),
    './air_data.csv',
]

df_raw = None
for p in CANDIDATE_PATHS:
    if os.path.exists(p):
        df_raw = pd.read_csv(p, encoding='utf-8')
        print(f"✓ 加载成功：{p}  （{df_raw.shape[0]} 行 × {df_raw.shape[1]} 列）")
        break

if df_raw is None:
    raise FileNotFoundError("未找到 air_data.csv，请将文件放在脚本同目录下")

print(f"\n原始数据概览：")
print(df_raw.describe().T[['count', 'mean', 'std', 'min', 'max']].head(10))

# ══════════════════════════════════════════════════════════════════════════
# 2. 数据清洗
# ══════════════════════════════════════════════════════════════════════════
print("\n【第二步】数据清洗")
print("-" * 70)

df = df_raw.copy()
n_before = len(df)

# 2.1 删除关键字段缺失行（票价、折扣率、飞行里程）
df.dropna(subset=['SUM_YR_1', 'SUM_YR_2', 'avg_discount', 'SEG_KM_SUM'], inplace=True)
print(f"  删除关键字段缺失行后：{n_before} → {len(df)}")

# 2.2 丢弃票价为0但平均折扣率不为0或总里程大于0的异常数据
#     （票价为0说明无效订单；但折扣率>0或里程>0则存在矛盾）
mask_abnormal = (
    ((df['SUM_YR_1'] == 0) & (df['SUM_YR_2'] == 0)) &
    ((df['avg_discount'] != 0) | (df['SEG_KM_SUM'] > 0))
)
n_before2 = len(df)
df = df[~mask_abnormal]
print(f"  删除票价异常行后：{n_before2} → {len(df)}")

# 2.3 删除总飞行里程为0的记录
n_before3 = len(df)
df = df[df['SEG_KM_SUM'] > 0]
print(f"  删除零里程行后：{n_before3} → {len(df)}")

# 2.4 删除折扣率异常（>1.5 或 <0）的记录
n_before4 = len(df)
df = df[(df['avg_discount'] >= 0) & (df['avg_discount'] <= 1.5)]
print(f"  删除折扣率异常行后：{n_before4} → {len(df)}")

print(f"\n  最终有效样本数：{len(df)}")

# ══════════════════════════════════════════════════════════════════════════
# 3. LRFMC 特征构建
# ══════════════════════════════════════════════════════════════════════════
print("\n【第三步】LRFMC 特征构建")
print("-" * 70)

# 以数据集中最大 LOAD_TIME 作为观测时间节点
df['LOAD_TIME'] = pd.to_datetime(df['LOAD_TIME'], format='mixed', dayfirst=False, errors='coerce')
df['FFP_DATE']  = pd.to_datetime(df['FFP_DATE'],  format='mixed', dayfirst=False, errors='coerce')
df['LAST_FLIGHT_DATE'] = pd.to_datetime(df['LAST_FLIGHT_DATE'], format='mixed', dayfirst=False, errors='coerce')

# 删除日期解析失败的行
df.dropna(subset=['LOAD_TIME', 'FFP_DATE', 'LAST_FLIGHT_DATE'], inplace=True)

load_time_max = df['LOAD_TIME'].max()
print(f"  观测时间节点（LOAD_TIME最大值）：{load_time_max.date()}")

lrfmc = pd.DataFrame()

# L：入会时间长度 = 观测时间 - 入会时间（月数）
lrfmc['L'] = ((load_time_max - df['FFP_DATE']).dt.days / 30).values

# R：最近消费时间间隔 = 观测时间 - 最近飞行日期（天数）
lrfmc['R'] = ((load_time_max - df['LAST_FLIGHT_DATE']).dt.days).values

# F：消费频率 = 观测窗口内飞行次数
lrfmc['F'] = df['FLIGHT_COUNT'].values

# M：飞行里程 = 总飞行公里数
lrfmc['M'] = df['SEG_KM_SUM'].values

# C：平均折扣率
lrfmc['C'] = df['avg_discount'].values

print("\n  LRFMC 特征统计：")
print(lrfmc.describe().round(2))

# ══════════════════════════════════════════════════════════════════════════
# 4. 数据标准化
# ══════════════════════════════════════════════════════════════════════════
print("\n【第四步】数据标准化（StandardScaler）")
print("-" * 70)

scaler = StandardScaler()
lrfmc_scaled = scaler.fit_transform(lrfmc)
lrfmc_scaled_df = pd.DataFrame(lrfmc_scaled, columns=['L', 'R', 'F', 'M', 'C'])

print("  标准化后各特征统计（均值≈0，标准差≈1）：")
print(lrfmc_scaled_df.describe().round(4))

# ══════════════════════════════════════════════════════════════════════════
# 5. 确定最优 K（肘部法则 + 轮廓系数）
# ══════════════════════════════════════════════════════════════════════════
print("\n【第五步】确定最优 K 值")
print("-" * 70)

K_RANGE = range(2, 9)
inertias, silhouettes = [], []

# 取样本子集加速计算（最多用 10000 条）
sample_size = min(10000, len(lrfmc_scaled))
np.random.seed(42)
sample_idx = np.random.choice(len(lrfmc_scaled), sample_size, replace=False)
X_sample = lrfmc_scaled[sample_idx]

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    km.fit(X_sample)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_sample, km.labels_, sample_size=min(3000, sample_size))
    silhouettes.append(sil)
    print(f"  K={k}  Inertia={km.inertia_:.2f}  Silhouette={sil:.4f}")

# ══════════════════════════════════════════════════════════════════════════
# 6. K-Means 聚类（K=5）
# ══════════════════════════════════════════════════════════════════════════
print("\n【第六步】K-Means 聚类（K=5）")
print("-" * 70)

K = 5
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10, max_iter=500)
kmeans.fit(lrfmc_scaled)
labels = kmeans.labels_

lrfmc['cluster'] = labels
lrfmc_scaled_df['cluster'] = labels

# 聚类中心（还原到原始尺度）
centers_scaled = kmeans.cluster_centers_
centers_original = scaler.inverse_transform(centers_scaled)
centers_df = pd.DataFrame(centers_original, columns=['L', 'R', 'F', 'M', 'C'])
centers_df.index.name = '簇编号'
centers_df['客户数量'] = pd.Series(labels).value_counts().sort_index().values

print("\n  聚类中心（原始尺度）：")
print(centers_df.round(4))

# 各簇规模
cluster_counts = pd.Series(labels).value_counts().sort_index()
print("\n  各簇客户数量：")
for i, cnt in cluster_counts.items():
    print(f"    簇 {i}：{cnt} 人  ({cnt/len(labels)*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════
# 7. 客户价值分类与命名
# ══════════════════════════════════════════════════════════════════════════
print("\n【第七步】客户价值分类")
print("-" * 70)

# 按 F（消费频率）和 M（飞行里程）综合排序确定价值等级
# 高F+高M = 高价值；低F+低M = 低价值；R越小越活跃
centers_df['综合得分'] = (
    centers_df['F'] * 0.3 +
    centers_df['M'] * 0.3 / 10000 +
    centers_df['C'] * 0.2 * 100 +
    centers_df['L'] * 0.1 / 10 -
    centers_df['R'] * 0.1 / 30
)

# 根据中心特征自动打标签
# 规则：F最高 → 重要保持；R最小且F较高 → 重要发展；其余按得分
def label_cluster(row, centers):
    cluster_id = row.name
    f_rank = centers['F'].rank(ascending=False)[cluster_id]
    r_rank = centers['R'].rank(ascending=True)[cluster_id]   # R越小越好
    m_rank = centers['M'].rank(ascending=False)[cluster_id]
    score_rank = centers['综合得分'].rank(ascending=False)[cluster_id]

    if score_rank == 1:
        return '重要保持客户'
    elif score_rank == 2:
        return '重要发展客户'
    elif score_rank == 3:
        return '重要挽留客户'
    elif score_rank == 4:
        return '一般发展客户'
    else:
        return '低价值客户'

centers_df['客户类型'] = [label_cluster(centers_df.loc[i], centers_df)
                         for i in centers_df.index]

print("\n  客户类型划分：")
for i, row in centers_df.iterrows():
    print(f"    簇 {i} → {row['客户类型']}  "
          f"(L={row['L']:.0f}月, R={row['R']:.0f}天, "
          f"F={row['F']:.0f}次, M={row['M']:.0f}km, C={row['C']:.3f})")

# ══════════════════════════════════════════════════════════════════════════
# 8. 可视化
# ══════════════════════════════════════════════════════════════════════════
print("\n【第八步】可视化输出")
print("-" * 70)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
fig = plt.figure(figsize=(18, 12))
fig.suptitle('实验六：航空公司客户价值 K-Means 分析', fontsize=16, fontweight='bold', y=0.98)

# ── 子图1：肘部法则 ───────────────────────────────────────────────────────
ax1 = fig.add_subplot(2, 3, 1)
ax1.plot(list(K_RANGE), inertias, 'o-', color='#5B9BD5', linewidth=2, markersize=6)
ax1.axvline(x=5, color='red', linestyle='--', alpha=0.6, label='K=5')
ax1.set_xlabel('聚类数 K'); ax1.set_ylabel('SSE（惯性）')
ax1.set_title('肘部法则 (Elbow Method)', fontweight='bold')
ax1.legend(); ax1.grid(alpha=0.3)

# ── 子图2：轮廓系数 ──────────────────────────────────────────────────────
ax2 = fig.add_subplot(2, 3, 2)
ax2.plot(list(K_RANGE), silhouettes, 's-', color='#ED7D31', linewidth=2, markersize=6)
ax2.axvline(x=5, color='red', linestyle='--', alpha=0.6, label='K=5')
ax2.set_xlabel('聚类数 K'); ax2.set_ylabel('轮廓系数')
ax2.set_title('轮廓系数 (Silhouette Score)', fontweight='bold')
ax2.legend(); ax2.grid(alpha=0.3)

# ── 子图3：各簇客户数量 ──────────────────────────────────────────────────
ax3 = fig.add_subplot(2, 3, 3)
cluster_labels_named = [centers_df.loc[i, '客户类型'] for i in range(K)]
colors_pie = ['#5B9BD5', '#ED7D31', '#70AD47', '#FFC000', '#FF6B6B']
wedges, texts, autotexts = ax3.pie(
    cluster_counts.values,
    labels=cluster_labels_named,
    autopct='%1.1f%%',
    colors=colors_pie,
    startangle=90,
    textprops={'fontsize': 9}
)
ax3.set_title('各类客户占比分布', fontweight='bold')

# ── 子图4：雷达图（标准化聚类中心）──────────────────────────────────────
ax4 = fig.add_subplot(2, 3, 4, polar=True)

features = ['L（入会时长）', 'R（消费间隔）', 'F（飞行频率）', 'M（飞行里程）', 'C（平均折扣）']
N = len(features)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # 闭合

# 对标准化中心进行 Min-Max 归一化到 [0,1] 以便雷达图展示
centers_for_radar = pd.DataFrame(centers_scaled, columns=['L', 'R', 'F', 'M', 'C'])
for col in centers_for_radar.columns:
    col_min = centers_for_radar[col].min()
    col_max = centers_for_radar[col].max()
    denom = col_max - col_min if col_max != col_min else 1
    centers_for_radar[col] = (centers_for_radar[col] - col_min) / denom

for i in range(K):
    values = centers_for_radar.iloc[i].tolist()
    values += values[:1]
    ax4.plot(angles, values, 'o-', linewidth=2, color=colors_pie[i],
             label=centers_df.loc[i, '客户类型'])
    ax4.fill(angles, values, alpha=0.1, color=colors_pie[i])

ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(features, fontsize=9)
ax4.set_ylim(0, 1)
ax4.set_title('LRFMC 雷达图（各类客户特征）', fontweight='bold', pad=20)
ax4.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=8)
ax4.grid(True)

# ── 子图5：聚类中心热力图 ────────────────────────────────────────────────
ax5 = fig.add_subplot(2, 3, 5)
import matplotlib.cm as cm
heat_data = centers_for_radar.values
im = ax5.imshow(heat_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax5.set_xticks(range(N))
ax5.set_xticklabels(['L', 'R', 'F', 'M', 'C'], fontsize=11)
ax5.set_yticks(range(K))
ax5.set_yticklabels([centers_df.loc[i, '客户类型'] for i in range(K)], fontsize=9)
ax5.set_title('聚类中心热力图（归一化）', fontweight='bold')
plt.colorbar(im, ax=ax5, shrink=0.8)
for i in range(K):
    for j in range(N):
        ax5.text(j, i, f'{heat_data[i, j]:.2f}', ha='center', va='center',
                 fontsize=9, color='black' if heat_data[i, j] < 0.7 else 'white')

# ── 子图6：F vs M 散点图（按簇着色）──────────────────────────────────────
ax6 = fig.add_subplot(2, 3, 6)
sample_plot = min(5000, len(lrfmc))
idx_plot = np.random.choice(len(lrfmc), sample_plot, replace=False)
for i in range(K):
    mask = lrfmc['cluster'].iloc[idx_plot] == i
    ax6.scatter(
        lrfmc['F'].iloc[idx_plot][mask],
        lrfmc['M'].iloc[idx_plot][mask] / 10000,
        c=colors_pie[i], alpha=0.4, s=8,
        label=centers_df.loc[i, '客户类型']
    )
ax6.set_xlabel('飞行频率 F（次）')
ax6.set_ylabel('飞行里程 M（万km）')
ax6.set_title('客户分布：飞行频率 vs 里程', fontweight='bold')
ax6.legend(fontsize=8, markerscale=3)
ax6.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out_png = os.path.join(OUT_DIR, 'exp6_kmeans_results.png')
plt.savefig(out_png, dpi=200, bbox_inches='tight')
print(f"✓ 图表已保存：{out_png}")
plt.show()

# ══════════════════════════════════════════════════════════════════════════
# 9. 汇总输出
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("实验总结")
print("=" * 70)
print(f"  有效样本数   ：{len(lrfmc)}")
print(f"  聚类数 K     ：{K}")
print(f"  最终SSE      ：{kmeans.inertia_:.2f}")
print(f"  轮廓系数     ：{silhouette_score(lrfmc_scaled[sample_idx], kmeans.labels_[sample_idx], sample_size=3000):.4f}")
print("\n  聚类中心（原始尺度）：")
print(centers_df[['L', 'R', 'F', 'M', 'C', '客户数量', '客户类型']].round(2).to_string())
print("=" * 70)
