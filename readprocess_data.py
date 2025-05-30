import scanpy as sc
import argparse
import numpy as np

# 用于查看已经预处理的数据列名，终端反馈 
parser = argparse.ArgumentParser(description='处理单细胞数据')
parser.add_argument('-i', '--input', type=str, required=True,
                    help='输入的h5ad文件路径')
args = parser.parse_args()

# 读取数据
adata = sc.read_h5ad(args.input)

print(adata.obs.columns) #看列名

def check_data(h5ad_file):
    """检查H5AD文件中的基因表达数据"""
    adata = sc.read_h5ad(h5ad_file)
    
    print("数据基本信息:")
    print(f"细胞数: {adata.n_obs}")
    print(f"基因数: {adata.n_vars}")
    
    key_genes = ['FOXP3', 'IL2RA', 'BECN1']
    print("\n关键基因是否存在:")
    for gene in key_genes:
        present = gene in adata.var_names
        if present:
            expr = adata[:, gene].X.toarray().flatten()
            n_expressed = np.sum(expr > 0)
            print(f"{gene}: 存在, {n_expressed}个细胞有表达 ({n_expressed/len(expr)*100:.2f}%)")
        else:
            print(f"{gene}: 不存在")
    
    print("\n基因名称示例:")
    print(list(adata.var_names[:10]))
    
    return adata

input_file = args.input
check_data(input_file)

