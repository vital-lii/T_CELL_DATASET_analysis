import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
from scipy import stats
import argparse

# 设置命令行参数
parser = argparse.ArgumentParser(description='分析CD4 Treg细胞中的BECN1表达')
parser.add_argument('-i', '--input', type=str, required=True, 
                    help='输入的CD4_Treg_analyzed.h5ad文件路径')
parser.add_argument('-o', '--output_dir', type=str, required=True, 
                    help='输出目录路径')
parser.add_argument('-g', '--gene', type=str, default='BECN1',
                    help='要分析的基因，默认为BECN1')
args = parser.parse_args()

# 创建输出目录
os.makedirs(args.output_dir, exist_ok=True)

# 读取数据
print(f"读取数据: {args.input}")
adata = sc.read_h5ad(args.input)
print(f"数据形状: {adata.shape}")

# 检查CD4 Treg列是否存在
if 'is_CD4_Treg' not in adata.obs.columns:
    print("错误: 输入数据中没有'is_CD4_Treg'列！")
    exit(1)

# 统计CD4 Treg细胞数量
n_treg = adata.obs['is_CD4_Treg'].sum()
print(f"CD4 Treg细胞数量: {n_treg} ({n_treg/len(adata)*100:.2f}%)")

# 检查目标基因是否存在
target_gene = args.gene
if target_gene not in adata.var_names:
    print(f"错误: 输入数据中没有'{target_gene}'基因！")
    exit(1)

print(f"\n分析{target_gene}在CD4 Treg细胞中的表达情况...")

# 1. 计算基因在CD4 Treg和非CD4 Treg细胞中的平均表达
gene_expr = adata[:, target_gene].X.toarray().flatten() if hasattr(adata[:, target_gene].X, 'toarray') else adata[:, target_gene].X.flatten()
treg_expr = gene_expr[adata.obs['is_CD4_Treg']]
non_treg_expr = gene_expr[~adata.obs['is_CD4_Treg']]

# 计算统计信息
treg_mean = np.mean(treg_expr)
non_treg_mean = np.mean(non_treg_expr)
treg_median = np.median(treg_expr)
non_treg_median = np.median(non_treg_expr)
treg_nonzero = np.sum(treg_expr > 0) / len(treg_expr) * 100
non_treg_nonzero = np.sum(non_treg_expr > 0) / len(non_treg_expr) * 100

print(f"{target_gene}表达统计信息:")
print(f"  CD4 Treg细胞 (n={len(treg_expr)}):")
print(f"    平均表达: {treg_mean:.4f}")
print(f"    中位数表达: {treg_median:.4f}")
print(f"    有表达的细胞比例: {treg_nonzero:.2f}%")
print(f"  非CD4 Treg细胞 (n={len(non_treg_expr)}):")
print(f"    平均表达: {non_treg_mean:.4f}")
print(f"    中位数表达: {non_treg_median:.4f}")
print(f"    有表达的细胞比例: {non_treg_nonzero:.2f}%")

# 2. 统计检验
print("\n执行统计检验...")
# 标准检验(使用全部数据)
t_stat, p_val_t = stats.ttest_ind(treg_expr, non_treg_expr, equal_var=False)
u_stat, p_val_u = stats.mannwhitneyu(treg_expr, non_treg_expr)
print(f"  全部数据 - Welch's t-test: t={t_stat:.4f}, p={p_val_t:.6f}")
print(f"  全部数据 - Mann-Whitney U test: U={u_stat:.4f}, p={p_val_u:.6f}")

# 添加下采样检验
n_samples = 100  # 重复采样次数
n_treg = len(treg_expr)  # CD4 Treg组样本数
tstat_samples = []
pval_samples = []

print("\n执行下采样检验(平衡样本量)...")
for i in range(n_samples):
    # 随机抽取与CD4 Treg组相同数量的非CD4 Treg细胞
    sample_indices = np.random.choice(len(non_treg_expr), size=n_treg, replace=False)
    sampled_non_treg = non_treg_expr[sample_indices]
    
    # 使用平衡样本量进行t检验
    t_stat_sample, p_val_sample = stats.ttest_ind(treg_expr, sampled_non_treg, equal_var=False)
    tstat_samples.append(t_stat_sample)
    pval_samples.append(p_val_sample)
    
    # 每20次报告进度
    if (i+1) % 20 == 0:
        print(f"  已完成 {i+1}/{n_samples} 次采样")

# 计算下采样结果的汇总统计
mean_tstat = np.mean(tstat_samples)
mean_pval = np.mean(pval_samples)
median_pval = np.median(pval_samples)
sig_ratio = sum(p < 0.05 for p in pval_samples) / n_samples

print(f"\n下采样统计结果 ({n_samples}次):")
print(f"  平均t统计量: {mean_tstat:.4f}")
print(f"  平均p值: {mean_pval:.6f}")
print(f"  中位数p值: {median_pval:.6f}")
print(f"  显著结果比例(p<0.05): {sig_ratio*100:.1f}%")

# 计算效应量(Cohen's d)
mean1, mean2 = np.mean(treg_expr), np.mean(non_treg_expr)
std1, std2 = np.std(treg_expr), np.std(non_treg_expr)
n1, n2 = len(treg_expr), len(non_treg_expr)
pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
cohens_d = (mean1 - mean2) / pooled_std

print(f"\n效应量:")
print(f"  Cohen's d: {cohens_d:.4f}")
print(f"  解释: |d|>0.8为大效应, |d|>0.5为中等效应, |d|>0.2为小效应")

# 将统计结果保存到文件
stats_df = pd.DataFrame({
    'Statistic': ['CD4 Treg Mean', 'Non-CD4-Treg Mean', 'CD4 Treg Median', 'Non-CD4-Treg Median', 
                 'CD4 Treg % Expressed', 'Non-CD4-Treg % Expressed', 't-statistic', 'p-value (t-test)', 
                 'U-statistic', 'p-value (Mann-Whitney)'],
    'Value': [treg_mean, non_treg_mean, treg_median, non_treg_median, 
             treg_nonzero, non_treg_nonzero, t_stat, p_val_t, u_stat, p_val_u]
})
stats_df.to_csv(os.path.join(args.output_dir, f'{target_gene}_expression_stats.csv'), index=False)

# 3. 绘图
print("\n生成可视化图表...")

# 创建小提琴图
plt.figure(figsize=(10, 8))
# 准备数据
violin_data = pd.DataFrame({
    target_gene: np.concatenate([treg_expr, non_treg_expr]),
    'Cell Type': ['CD4 Treg'] * len(treg_expr) + ['Non-CD4-Treg'] * len(non_treg_expr)
})

# 绘制小提琴图
ax = sns.violinplot(x='Cell Type', y=target_gene, data=violin_data, 
                   palette={'CD4 Treg': '#FF7F0E', 'Non-CD4-Treg': '#1F77B4'})
# 添加点以显示均值
plt.plot([0, 1], [treg_mean, non_treg_mean], 'ko', markersize=8)
# 添加p值标记
max_val = max(np.percentile(treg_expr, 99), np.percentile(non_treg_expr, 99))
y_pos = max_val * 1.1
if p_val_t < 0.001:
    plt.text(0.5, y_pos, '***', ha='center', va='bottom', fontsize=20)
elif p_val_t < 0.01:
    plt.text(0.5, y_pos, '**', ha='center', va='bottom', fontsize=20)
elif p_val_t < 0.05:
    plt.text(0.5, y_pos, '*', ha='center', va='bottom', fontsize=20)
else:
    plt.text(0.5, y_pos, 'ns', ha='center', va='bottom', fontsize=14)

plt.title(f'{target_gene} Expression in CD4 Treg vs Non-CD4-Treg Cells\np={p_val_t:.2e}')
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, f'{target_gene}_expression_violin.png'), dpi=300)
print(f"已保存小提琴图到: {os.path.join(args.output_dir, f'{target_gene}_expression_violin.png')}")
plt.close()

# 绘制箱线图
plt.figure(figsize=(10, 8))
ax = sns.boxplot(x='Cell Type', y=target_gene, data=violin_data, 
                palette={'CD4 Treg': '#FF7F0E', 'Non-CD4-Treg': '#1F77B4'})
# 在箱线图上加上散点图
sns.stripplot(x='Cell Type', y=target_gene, data=violin_data, size=3, alpha=0.3, 
              jitter=True, color='black')
plt.title(f'{target_gene} Expression in CD4 Treg vs Non-CD4-Treg Cells\np={p_val_t:.2e}')
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, f'{target_gene}_expression_boxplot.png'), dpi=300)
print(f"已保存箱线图到: {os.path.join(args.output_dir, f'{target_gene}_expression_boxplot.png')}")
plt.close()

# 如果UMAP坐标存在，绘制UMAP上的表达分布
if 'X_umap' in adata.obsm:
    print("\n绘制UMAP上的表达分布...")
    
    # 绘制目标基因在UMAP上的表达
    plt.figure(figsize=(12, 10))
    sc.pl.umap(adata, color=target_gene, cmap='viridis', size=20, show=False)
    plt.title(f'{target_gene} Expression')
    plt.savefig(os.path.join(args.output_dir, f'{target_gene}_umap_expression.png'), dpi=300, bbox_inches='tight')
    print(f"已保存UMAP表达图到: {os.path.join(args.output_dir, f'{target_gene}_umap_expression.png')}")
    plt.close()
    
    # 绘制CD4 Treg和目标基因共表达的UMAP图
    # 创建一个描述性的类别变量
    gene_high_expr = gene_expr > np.percentile(gene_expr, 75)  # 取前25%为高表达
    is_treg = adata.obs['is_CD4_Treg'].astype(bool)
    
    # 创建带有明确标签的类别变量
    condition = pd.Series(["Other" for _ in range(len(adata))], index=adata.obs.index)
    condition[is_treg & gene_high_expr] = f"High {target_gene} CD4 Treg"
    adata.obs[f'{target_gene}_expression_group'] = condition
    
    # 绘制标记图
    plt.figure(figsize=(12, 10))
    sc.pl.umap(adata, color=f'{target_gene}_expression_group', 
               title=f'CD4 Treg cells with high {target_gene}',
               palette={f"High {target_gene} CD4 Treg": 'red', "Other": 'lightgrey'},
               size=20, show=False)
    plt.savefig(os.path.join(args.output_dir, f'high_{target_gene}_treg_umap.png'), dpi=300, bbox_inches='tight')
    print(f"已保存高表达{target_gene}的CD4 Treg细胞UMAP图到: {os.path.join(args.output_dir, f'high_{target_gene}_treg_umap.png')}")
    plt.close()

# 4. 全基因差异表达分析（找出CD4 Treg中差异表达基因）
print("\n执行CD4 Treg vs 非CD4 Treg的差异表达分析...")
# 使用is_CD4_Treg作为分组
adata.obs['treg_group'] = adata.obs['is_CD4_Treg'].astype(str)
sc.tl.rank_genes_groups(adata, 'treg_group', method='wilcoxon')

# 提取结果
result = sc.get.rank_genes_groups_df(adata, group='True')  # CD4 Treg vs 非CD4 Treg
result = result.sort_values('pvals_adj')  # 按校正p值排序

# 保存结果
result.to_csv(os.path.join(args.output_dir, 'CD4_Treg_vs_non_DEGs.csv'), index=False)
print(f"已保存差异表达基因结果到: {os.path.join(args.output_dir, 'CD4_Treg_vs_non_DEGs.csv')}")

# 绘制差异表达基因火山图
plt.figure(figsize=(12, 10))
# 计算-log10(p)和log2(FC)
result['-log10p'] = -np.log10(result['pvals'].astype(float))
result['log2fc'] = result['logfoldchanges'].astype(float)

# 标记显著差异和靶基因
result['significant'] = result['pvals_adj'] < 0.05
result['target'] = result['names'] == target_gene

# 绘制散点图
plt.scatter(result[~result['significant']]['log2fc'], 
            result[~result['significant']]['-log10p'], 
            color='lightgrey', alpha=0.6, s=20)
plt.scatter(result[result['significant']]['log2fc'], 
            result[result['significant']]['-log10p'], 
            color='blue', alpha=0.6, s=20)

# 特别标记目标基因
if target_gene in result['names'].values:
    target_data = result[result['names'] == target_gene]
    plt.scatter(target_data['log2fc'], target_data['-log10p'], 
                color='red', alpha=1.0, s=100, edgecolor='black')
    plt.text(target_data['log2fc'].values[0], target_data['-log10p'].values[0],
             target_gene, ha='right', fontsize=14)

# 添加参考线
plt.axhline(-np.log10(0.05), linestyle='--', color='gray')
plt.axvline(0, linestyle='--', color='gray')

plt.xlabel('Log2 Fold Change (CD4 Treg / Non-CD4-Treg)')
plt.ylabel('-Log10 P-value')
plt.title('Differentially Expressed Genes in CD4 Treg vs Non-CD4-Treg Cells')
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'CD4_Treg_DEG_volcano.png'), dpi=300)
print(f"已保存火山图到: {os.path.join(args.output_dir, 'CD4_Treg_DEG_volcano.png')}")
plt.close()

# 绘制热图显示top差异基因
top_genes = result.head(25)['names'].tolist()  # 取前25个差异基因
if target_gene not in top_genes and target_gene in adata.var_names:
    top_genes.append(target_gene)  # 确保目标基因包含在热图中

# 准备热图数据
if top_genes:
    plt.figure(figsize=(15, 10))
    sc.pl.heatmap(adata, top_genes, groupby='treg_group', 
                 dendrogram=True, standard_scale='var', 
                 show_gene_labels=True, show=False)
    plt.savefig(os.path.join(args.output_dir, 'CD4_Treg_DEG_heatmap.png'), dpi=300, bbox_inches='tight')
    print(f"已保存热图到: {os.path.join(args.output_dir, 'CD4_Treg_DEG_heatmap.png')}")
    plt.close()

print(f"\n分析完成！结果保存在: {args.output_dir}")
