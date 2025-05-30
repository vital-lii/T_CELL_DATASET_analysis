import pandas as pd
import os
import argparse

def extract_significant_genes(file_path, log_fc_threshold=1.0, p_value_threshold=0.05, output_file=None):
    """
    从差异表达基因CSV文件中提取满足条件的基因名
    
    参数:
    file_path: CSV文件路径
    log_fc_threshold: log fold change的绝对值阈值
    p_value_threshold: 调整后p值的阈值
    output_file: 输出文件路径，如果为None则不保存文件
    
    返回:
    满足条件的基因名列表
    """
    # 读取CSV文件
    print(f"读取文件: {file_path}")
    df = pd.read_csv(file_path)
    
    # 筛选满足条件的基因
    significant_genes = df[(abs(df['logfoldchanges']) > log_fc_threshold) & 
                          (df['pvals'] < p_value_threshold)]
    
    # 排序 - 先按p值升序，再按log fold change绝对值降序
    significant_genes = significant_genes.sort_values(
        by=['pvals', 'logfoldchanges'], 
        ascending=[True, False]
    )
    
    # 提取基因名
    gene_names = significant_genes['names'].tolist()
    
    # 打印统计信息
    print(f"总基因数: {len(df)}")
    print(f"满足条件的基因数 (|logFC|>{log_fc_threshold} & p<{p_value_threshold}): {len(gene_names)}")
    
    # 保存结果到文件
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 保存完整的筛选后数据框
        significant_genes.to_csv(output_file, index=False)
        print(f"已将满足条件的基因保存到: {output_file}")
        
        # 仅保存基因名列表
        gene_list_file = os.path.splitext(output_file)[0] + "_names_only.txt"
        with open(gene_list_file, 'w') as f:
            for gene in gene_names:
                f.write(f"{gene}\n")
        print(f"已将基因名列表保存到: {gene_list_file}")
    
    return gene_names

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='从差异表达基因CSV文件中提取满足条件的基因名')
    
    # 添加参数
    parser.add_argument('-i','--input_file', type=str, required=True, help='输入的CSV文件路径')
    parser.add_argument('-o','--output_file', type=str, required=True, help='输出的CSV文件路径')
    parser.add_argument('-l','--log_fc_threshold', type=float, default=1.0, help='log fold change的绝对值阈值')
    parser.add_argument('-p','--p_value_threshold', type=float, default=0.05, help='调整后p值的阈值')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 提取显著差异基因
    significant_genes = extract_significant_genes(
        file_path=args.input_file,
        log_fc_threshold=args.log_fc_threshold,
        p_value_threshold=args.p_value_threshold,
        output_file=args.output_file
    )
    
    # 打印前20个基因
    print("\n前20个显著差异基因:")
    for i, gene in enumerate(significant_genes[:20], 1):
        print(f"{i}. {gene}")
