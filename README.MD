# 单细胞数据分析- BECN1 与TREG相关研究

数据名称:人类 T 细胞的单细胞转录组学揭示了健康和疾病中的组织和激活特征  （24-5-29更新）

网址：[Single-cell transcriptomics of human T cells reveals tissue and activation signatures in health and disease - Overview - HCA Data Explorer](https://explore.data.humancellatlas.org/projects/4a95101c-9ffc-4f30-a809-f04518a23803)

数据集：lung & blood (2个)

## 1.loom文件转换为h5ad文件

loadloom.py, 指定-i 和 -o 带有后缀H5AD的输出文件名（自建后缀）
用于后续分析

## 2. 使用scanpy进行数据预处理

若文件太大，去除无意义的。用preprocess.py 先处理一步。

* 参数说明：
  -i, --input：输入H5AD文件路径
  -o, --output_dir：输出目录
  --min_genes：每个细胞的最小基因数 (默认: 200)
  --min_cells：每个基因的最小细胞数 (默认: 3)
  --max_mt_pct：最大线粒体基因百分比 (默认: 20)
  --show_plots：显示图片弹窗而不仅是保存 (可选)

**目的：**
 1.过滤低质量细胞（基因数<200, 线粒体比例>20%，若线粒体比例过高，可能细胞已凋亡，胞质mRNA相对不稳定可能已经被降解，意义不大，且一般哺乳动物细胞中的线粒体基因占比5-15%，若为高代谢细胞如心肌细胞等，可能略高，但<20%还是较为合理）
2. 过滤低表达基因（每个细胞中表达的基因数<3）
生成QC指标图表（QC_METRICS.PNG）

2.数据标准化：
总计数标准化 (target_sum=1e4)
对数转换

3.特征选择
选择高变异基因
生成高变异基因图表 (highly_variable_genes.png)

4.降维与聚类
PCA降维 (50个主成分)
UMAP降维 (2维可视化)
Leiden聚类 (9个聚类, 0-8)
生成聚类图表 (umap_clusters.png)

5.标记基因分析
识别每个聚类的标记基因
生成标记基因热图 (marker_genes.png)

6.关键基因分析：
分析BECN1, FOXP3, CD4, CD8A, IL2RA, CTLA4等关键基因
生成关键基因UMAP图 (key_genes_umap.png)
生成关键基因小提琴图 (key_genes_violin.png)

7.Treg细胞识别(单个依据FOXP3进行))
基于：
基于FOXP3高表达识别Treg细胞
比较Treg和非Treg细胞中BECN1表达
生成Treg细胞分布图 (treg_cells_umap.png)
生成BECN1-FOXP3相关性图 (becn1_foxp3_correlation.png)

**结果：**

*lung_result:*
1.T细胞亚群分类：

* 识别9个亚群（0-8）
* 聚类6和聚类8表现出明显的Treg细胞特征 (高表达FOXP3, IL2RA, CTLA4)

2.BECN1与Treg细胞关系（可能，因为TREG的定义只是依据FOXP3的表达量进行，表达前10的定义为TREG）
终端反馈：Treg细胞中BECN1平均表达显著高于非Treg细胞 (0.277 vs 0.125, p=5.14e-16)
聚类6和聚类8中BECN1表达水平较高
BECN1与FOXP3表达存在一定相关性：图becn1_foxp3_correlation.png为证

*Blood_result*
共10个聚类，其中8.2.9可能为TREG，且BECN1表达较高

但结合key_genes_violin.png以及其他图来看，更有可能是2

可能：

Treg细胞中BECN1平均表达: 0.23633956909179688
非Treg细胞中BECN1平均表达: 0.15695540606975555
BECN1在Treg vs 非Treg细胞中的t检验: t=6.374860823787254, p=1.870071496598315e-10

## 3.后续分析：精确定义CD4 Treg细胞

**数据内容：标记了多个基因（至少三个）**

* processed_data.h5ad ： 预处理后的数据，可用用readprocess_data.py来查看列名
* 使用distin.py来精准定义，并添加is_CD4_Treg标记列
* 可视化CD4treg分布


输出文件：

* CD4_Treg_analyzed.h5ad（含有CD4 Treg标记的数据）
* CD4_Treg_umap.png（CD4 Treg细胞分布图）
* CD4_Treg_markers_umap.png（标记基因表达图）

## 4.然后再cd4_treg_analyzed.h5ad中，

* 使用该列名下的CD4 Treg细胞进行基因表达差异分析
  用generelate.py 进行分析 注意短参数命令可以指定基因名称 输出文件：BECN1_expression_violin.png

肺部T细胞数据集最终结果解释：BECN1在CD4 Treg细胞中的表达显著高于非CD4 Treg细胞（p<0.001），表现为中小程度的效应（Cohen's d=0.43）
虽然表达水平差异的幅度适中，但CD4 Treg中有表达的细胞比例（39.17%）远高于非CD4 Treg细胞（8.09%）

## 5.提取cd4_treg 与非cd4_treg 的差异基因。
来源：CD4_Treg_vs_non_DEGs.csv

工具：extrac.py 提取logfc绝对值大于1，p值小于0.05的基因。

输出文件：extract.csv(自建空白文件)

## 6.差异基因富集
enrich_mul.py 富集分析
