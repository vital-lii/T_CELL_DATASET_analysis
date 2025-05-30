import numpy as np
import scanpy as sc
import argparse
import os
import matplotlib.pyplot as plt  

# 以markers找CD4 Treg 
def check_treg_markers(adata):
    """检查CD4 Treg相关标记基因的可用性和表达情况"""
    # CD4 Treg相关标记
    treg_markers = [
        'FOXP3',  # 核心转录因子
        'IL2RA',  # CD25
        'CTLA4',  # 抑制性受体
        'TNFRSF18',  # 好像外周血的Treg会表达  来自原文
        'TIGIT',
        'TNFRSF4',
        'PMCH'
    ]
    
    # 检查哪些标记在数据中可用
    available_markers = [m for m in treg_markers if m in adata.var_names]
    print("可用的CD4 Treg标记基因:")
    
    # 统计每个标记的表达情况
    for marker in available_markers:
        expr = adata[:, marker].X.toarray().flatten()
        n_expressed = np.sum(expr > 0)
        percent = n_expressed / len(expr) * 100
        print(f"{marker}: {n_expressed} cells ({percent:.1f}%)")
    
    return available_markers

def analyze_CD4_TREG_markers(adata):
    """分析CD4 Treg相关标记基因的表达分布"""
    treg_markers = ['FOXP3', 'IL2RA', 'CTLA4', 'TIGIT', 'PMCH']
    
    print("\nCD4 Treg标记基因表达情况:")
    for marker in treg_markers:
        if marker in adata.var_names:
            expr = adata[:, marker].X.toarray().flatten()
            n_expressed = np.sum(expr > 0)
            mean_expr = np.mean(expr[expr > 0])
            print(f"{marker}:")
            print(f"  表达细胞数: {n_expressed} ({n_expressed/len(adata)*100:.1f}%)")
            print(f"  平均表达量: {mean_expr:.2f}")
    
    return adata

def identify_CD4_Treg(adata, min_markers=3):
    """识别CD4 Treg细胞
    
    Parameters:
    -----------
    adata : AnnData
        包含单细胞数据的AnnData对象
    min_markers : int, default=3
        定义为Treg所需的最少标记基因数量
        
    Returns:
    --------
    adata : AnnData
        添加了'is_CD4_Treg'列的AnnData对象
    """
    treg_markers = ['FOXP3', 'IL2RA', 'CTLA4', 'TIGIT', 'PMCH']
    
    # 检查可用的标记
    available_markers = [m for m in treg_markers if m in adata.var_names]
    if len(available_markers) == 0:
        print("错误: 未找到任何Treg标记基因!")
        return adata
    
    print(f"可用的CD4 Treg标记基因: {available_markers}")
    
    # 计算每个标记的表达情况
    marker_expression = []
    for marker in available_markers:
        expr = adata[:, marker].X.toarray().flatten()
        marker_expression.append(expr > 0)
    
    # 计算每个细胞表达的标记数量
    n_markers_expressed = np.sum(marker_expression, axis=0)
    
    # 添加到adata.obs
    adata.obs['CD4_Treg_score'] = n_markers_expressed
    adata.obs['is_CD4_Treg'] = n_markers_expressed >= min_markers
    
    # 统计结果
    n_treg = adata.obs['is_CD4_Treg'].sum()
    print(f"\nCD4 Treg细胞识别结果 (至少表达{min_markers}个标记):")
    print(f"找到 {n_treg} 个CD4 Treg细胞 ({n_treg/len(adata)*100:.2f}%)")
    
    return adata

def plot_treg_distribution(adata, output_dir):
    """绘制CD4 Treg细胞在UMAP图上的分布"""
    print("\n绘制CD4 Treg细胞分布...")
    
    # 检查是否已有UMAP坐标
    if 'X_umap' in adata.obsm:
        print("发现现有UMAP坐标，直接使用...")
    else:
        print("未找到UMAP坐标，正在计算...")
        # 确保数据已标准化和对数转换
        if 'highly_variable' not in adata.var.columns:
            print("选择高变异基因...")
            sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        
        print("执行PCA...")
        sc.pp.pca(adata, n_comps=30, use_highly_variable=True)
        print("计算邻居图...")
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)
        print("执行UMAP降维...")
        sc.tl.umap(adata)
    
    # 创建更好的图形
    plt.figure(figsize=(12, 10))
    
    # 绘制UMAP图
    sc.pl.umap(adata, color=['is_CD4_Treg'], 
               title='CD4 Treg distribution',
               legend_loc='right',
               legend_fontsize=14,
               palette={'True': '#FF7F0E', 'False': '#1F77B4'},  # 自定义颜色
               legend_fontoutline=2,
               frameon=False,
               size=20,
               show=False)
    
    # 修改图例标签
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    new_labels = ['non-CD4-Treg', 'CD4 Treg']  
    ax.legend(handles, new_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
    
    # 调整布局确保图例完整显示
    plt.tight_layout(rect=[0, 0, 0.85, 1])  
    
    # 保存高质量图片
    save_path = os.path.join(output_dir, 'CD4_Treg_umap.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"已保存图片到: {save_path}")
    plt.close()
    
    # 绘制标记基因表达图
    treg_markers = ['FOXP3', 'IL2RA', 'CTLA4', 'TIGIT', 'PMCH']
    available_markers = [m for m in treg_markers if m in adata.var_names]
    
    if available_markers:
        plt.figure(figsize=(20, 16))  # 更大的图形尺寸
        sc.pl.umap(adata, color=available_markers, 
                   ncols=2, 
                   wspace=0.4,        # 增加子图水平间距
                   hspace=0.4,        # 增加子图垂直间距
                   cmap='viridis',
                   size=30,           # 增大点的大小
                   use_raw=False,
                   legend_loc='right',  # 图例放在右侧
                   legend_fontsize=12,
                   title=[f'{m} Expression' for m in available_markers],  # 每个子图的标题
                   show=False)
        
        # 调整整体布局
        plt.tight_layout()
        
        # 保存高质量图像
        save_path = os.path.join(output_dir, 'CD4_Treg_markers_umap.png')
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        print(f"已保存标记基因表达图到: {save_path}")
        plt.close()

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='分析CD4 Treg细胞')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='输入的h5ad文件路径')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='输出目录路径')
    parser.add_argument('-m', '--min_markers', type=int, default=3,
                        help='定义CD4 Treg所需的最少标记基因数量，默认为3')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 读取数据
    print(f"读取数据: {args.input}")
    adata = sc.read_h5ad(args.input)
    
    # 处理重复的索引
    print("处理重复的索引...")
    adata.obs_names_make_unique()  # 使细胞名称唯一
    adata.var_names_make_unique()  # 使基因名称唯一
    
    # 1. 分析CD4 Treg标记基因的表达
    print("\n分析CD4 Treg标记基因表达...")
    analyze_CD4_TREG_markers(adata)
    
    # 2. 识别CD4 Treg细胞
    print("\n识别CD4 Treg细胞...")
    adata = identify_CD4_Treg(adata, min_markers=args.min_markers)
    
    # 3. 绘制CD4 Treg分布图
    plot_treg_distribution(adata, args.output_dir)
    
    # 保存结果
    output_file = os.path.join(args.output_dir, 'CD4_Treg_analyzed.h5ad')
    print(f"\n保存结果到: {output_file}")
    adata.write(output_file)

if __name__ == "__main__":
    main()