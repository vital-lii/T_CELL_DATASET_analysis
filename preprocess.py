import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import matplotlib

matplotlib.use('Agg')

parser = argparse.ArgumentParser(description='预处理和分析单细胞数据')
parser.add_argument('-i', '--input', help='输入H5AD文件路径', required=True)
parser.add_argument('-o', '--output_dir', help='输出目录', required=True)
parser.add_argument('--min_genes', type=int, default=200, help='每个细胞的最小基因数')
parser.add_argument('--min_cells', type=int, default=3, help='每个基因的最小细胞数')
parser.add_argument('--max_mt_pct', type=float, default=20, help='最大线粒体基因百分比')
parser.add_argument('--show_plots', action='store_true', 
                    help='显示图片弹窗而不仅是保存')
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
print(f"输出目录: {os.path.abspath(args.output_dir)}")

try:
    test_file = os.path.join(args.output_dir, 'test_write.txt')
    with open(test_file, 'w') as f:
        f.write('测试写入权限')
    os.remove(test_file)
    print("输出目录写入权限正常")
except Exception as e:
    print(f"警告：输出目录可能没有写入权限: {str(e)}")

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=200, figsize=(10, 8))
sc.settings.figdir = args.output_dir
sc.settings.autoshow = False

print(f"加载数据: {args.input}")
adata = sc.read_h5ad(args.input)
print(f"数据形状: {adata.shape}")

print("检查基因名称是否唯一...")
if len(adata.var_names) != len(set(adata.var_names)):
    print(f"发现 {len(adata.var_names) - len(set(adata.var_names))} 个重复的基因名称，正在修复...")
    adata.var_names_make_unique()

print("计算质量控制指标...")
if 'mt' not in adata.var.columns:
    adata.var['mt'] = adata.var_names.str.startswith('MT-')

sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

print("绘制质量控制图...")
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
sns.histplot(adata.obs.n_genes_by_counts, kde=False, ax=axs[0])
axs[0].set_title('genes counts of cells')
sns.histplot(adata.obs.total_counts, kde=False, ax=axs[1])
axs[1].set_title('UMI counts of cells')
sns.histplot(adata.obs.pct_counts_mt, kde=False, ax=axs[2])
axs[2].set_title('percentage of mitochondrial genes')
sc.pl.highest_expr_genes(adata, n_top=20, ax=axs[3])
plt.tight_layout()
save_path = os.path.join(args.output_dir, 'qc_metrics.png')
plt.savefig(save_path)
print(f"已保存图片到: {save_path}")
plt.close(fig)

# 过滤低质量细胞和基因
print(f"过滤细胞 (最小基因数: {args.min_genes}, 最大线粒体百分比: {args.max_mt_pct})...")
adata = adata[adata.obs.n_genes_by_counts > args.min_genes, :]
adata = adata[adata.obs.pct_counts_mt < args.max_mt_pct, :]
print(f"过滤基因 (最小细胞数: {args.min_cells})...")
sc.pp.filter_genes(adata, min_cells=args.min_cells)
print(f"过滤后的数据形状: {adata.shape}")

# 2. 标准化和对数转换
print("标准化数据...")
sc.pp.normalize_total(adata, target_sum=1e4)
print("对数转换...")
sc.pp.log1p(adata)

print("选择高变异基因...")
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
print(f"高变异基因数量: {sum(adata.var.highly_variable)}")

var_genes = adata.var[adata.var.highly_variable].index.tolist()
with open(os.path.join(args.output_dir, 'highly_variable_genes.txt'), 'w') as f:
    for gene in var_genes:
        f.write(f"{gene}\n")

sc.pl.highly_variable_genes(adata, show=False)
save_path = os.path.join(args.output_dir, 'highly_variable_genes.png')
plt.savefig(save_path)
print(f"已保存图片到: {save_path}")
plt.close()

print("执行PCA...")
sc.pp.pca(adata, n_comps=50, use_highly_variable=True)
sc.pl.pca_variance_ratio(adata, n_pcs=50, show=False)
save_path = os.path.join(args.output_dir, 'pca_variance_ratio.png')
plt.savefig(save_path)
print(f"已保存图片到: {save_path}")
plt.close()

print("计算邻居图...")
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)

print("执行UMAP降维...")
sc.tl.umap(adata)
sc.pl.umap(adata, color=['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], show=False)
save_path = os.path.join(args.output_dir, 'umap_qc.png')
plt.savefig(save_path)
print(f"已保存图片到: {save_path}")
plt.close()

print("执行Leiden聚类...")
sc.tl.leiden(adata, resolution=0.5)

plt.figure(figsize=(14, 12))  # 更大的图形尺寸
ax = sc.pl.umap(adata, color=['leiden'], 
           legend_loc='right',  # 将图例放在右侧
           legend_fontsize=12,
           legend_fontoutline=2,
           frameon=False,
           title='Leiden Clustering',
           return_fig=True)  # 返回图形对象

plt.tight_layout(rect=[0, 0, 0.85, 1])

if args.show_plots:
    plt.show()

save_path = os.path.join(args.output_dir, 'umap_clusters.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"已保存图片到: {save_path}")
plt.close()

print("查找聚类标记基因...")
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, show=False)
save_path = os.path.join(args.output_dir, 'marker_genes.png')
plt.savefig(save_path)
print(f"已保存图片到: {save_path}")
plt.close()

key_genes = ['BECN1', 'FOXP3', 'CD4', 'CD8A', 'IL2RA', 'CTLA4','TIGIT','PMCH']
available_genes = [gene for gene in key_genes if gene in adata.var_names]

if available_genes:
    print(f"分析关键基因: {available_genes}")
    sc.pl.umap(adata, color=available_genes)
    plt.savefig(os.path.join(args.output_dir, 'key_genes_umap.png'))
    plt.close()
    
    sc.pl.violin(adata, available_genes, groupby='leiden')
    plt.savefig(os.path.join(args.output_dir, 'key_genes_violin.png'))
    plt.close()
    
    # 如果BECN1和FOXP3都存在，分析它们的共表达
    if 'BECN1' in available_genes and 'FOXP3' in available_genes:
        plt.figure(figsize=(8, 8))
        becn1_values = adata[:, 'BECN1'].X.toarray().flatten() if hasattr(adata[:, 'BECN1'].X, 'toarray') else adata[:, 'BECN1'].X.flatten()
        foxp3_values = adata[:, 'FOXP3'].X.toarray().flatten() if hasattr(adata[:, 'FOXP3'].X, 'toarray') else adata[:, 'FOXP3'].X.flatten()
        
        plt.scatter(becn1_values, foxp3_values, alpha=0.5, s=1)
        plt.xlabel('BECN1 expression')
        plt.ylabel('FOXP3 expression')
        plt.title('BECN1 vs FOXP3 expression')
        
        save_path = os.path.join(args.output_dir, 'becn1_foxp3_correlation.png')
        plt.savefig(save_path)
        print(f"已保存图片到: {save_path}")
        plt.close()
        
        # 识别可能的Treg细胞 (FOXP3高表达)
        if 'FOXP3' in available_genes:
            # 将稀疏矩阵转换为密集数组
            foxp3_values = adata[:, 'FOXP3'].X.toarray().flatten() if hasattr(adata[:, 'FOXP3'].X, 'toarray') else adata[:, 'FOXP3'].X.flatten()
            foxp3_threshold = np.percentile(foxp3_values, 90)
            adata.obs['is_treg'] = foxp3_values > foxp3_threshold
            
            plt.figure(figsize=(14, 12))
            
            sc.pl.umap(adata, color=['is_treg'], 
                       title='Treg distribution(based on FOXP3)',
                       legend_loc='right',
                       legend_fontsize=14,
                       palette={'True': '#FF7F0E', 'False': '#1F77B4'},  # 自定义颜色
                       legend_fontoutline=2,
                       frameon=False,
                       size=30,  # 增大点的大小
                       show=False)
            
            ax = plt.gca()
            handles, labels = ax.get_legend_handles_labels()
            new_labels = ['non-Treg', 'Treg']
            ax.legend(handles, new_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
            
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            
            save_path = os.path.join(args.output_dir, 'treg_cells_umap.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"已保存图片到: {save_path}")
            plt.close()
            
            # 比较Treg和非Treg细胞中BECN1的表达
            if 'BECN1' in available_genes:
                # 将稀疏矩阵转换为密集数组
                becn1_values = adata[:, 'BECN1'].X.toarray().flatten() if hasattr(adata[:, 'BECN1'].X, 'toarray') else adata[:, 'BECN1'].X.flatten()
                
                treg_becn1 = becn1_values[adata.obs['is_treg']]
                non_treg_becn1 = becn1_values[~adata.obs['is_treg']]
                
                print(f"Treg细胞中BECN1平均表达: {np.mean(treg_becn1)}")
                print(f"非Treg细胞中BECN1平均表达: {np.mean(non_treg_becn1)}")
                
                # 统计检验
                from scipy import stats
                t_stat, p_val = stats.ttest_ind(treg_becn1, non_treg_becn1)
                print(f"BECN1在Treg vs 非Treg细胞中的t检验: t={t_stat}, p={p_val}")
else:
    print(f"未找到关键基因。可用基因示例: {list(adata.var_names[:10])}")

print("保存处理后的数据...")
adata.write(os.path.join(args.output_dir, 'processed_data.h5ad'))

print(f"分析完成！结果保存在: {args.output_dir}")

if args.show_plots:
    plt.show()
