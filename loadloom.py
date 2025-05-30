import scanpy as sc
import loompy as lp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

parser = argparse.ArgumentParser(description='Process loom files')
parser.add_argument('-i', '--input', help='Input file path', required=True)
parser.add_argument('-o', '--output', help='Output file path', required=True)

args = parser.parse_args()

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=300, figsize=(8, 8))

file_path = args.input
output_path = args.output

adata = sc.read_loom(file_path)

print(f"数据形状: {adata.shape}")  # (细胞数, 基因数)
print(f"可用的基因名称: {list(adata.var_names[:10])}")  # 显示前10个基因名称
print(f"可用的变量: {list(adata.var.keys())}")  # 基因的元数据
print(f"可用的观察变量: {list(adata.obs.keys())}")  # 细胞的元数据

adata.write(output_path)
