# Single-cell Data Analysis - Study of BECN1 and Treg Correlation

**Dataset Name:** Single-cell transcriptomics of human T cells reveals tissue and activation signatures in health and disease (Updated: 24-05-29)  
**URL:** [Single-cell transcriptomics of human T cells reveals tissue and activation signatures in health and disease - Overview - HCA Data Explorer](https://explore.data.humancellatlas.org/projects/4a95101c-9ffc-4f30-a809-f04518a23803)  
**Datasets:** Lung & Blood (2 datasets)

## 1. Converting loom files to h5ad format

Using `loadloom.py` with parameters:  
`-i` input file  
`-o` output filename with `.h5ad` suffix (custom extension)  
For subsequent analysis

## 2. Data Preprocessing using Scanpy

For large files, preprocess with `preprocess.py` to remove insignificant data.

* **Parameters:**
  - `-i, --input`: Input H5AD file path
  - `-o, --output_dir`: Output directory
  - `--min_genes`: Minimum genes per cell (default: 200)
  - `--min_cells`: Minimum cells per gene (default: 3)
  - `--max_mt_pct`: Maximum mitochondrial gene percentage (default: 20)

**Objectives:**
1. Filter low-quality cells:
   - Genes < 200
   - Mitochondrial ratio > 20% (High mt% indicates apoptotic cells with degraded cytoplasmic mRNA. Mammalian cells typically have 5-15% mt genes; <20% is reasonable even for high-metabolism cells)
2. Filter low-expression genes:
   - Genes expressed in <3 cells
3. Generate QC metrics plot (`qc_metrics.png`)

**Processing Steps:**
1. Data normalization:
   - Total count normalization (`target_sum=1e4`)
   - Log transformation
2. Feature selection:
   - Identify highly variable genes
   - Generate plot (`highly_variable_genes.png`)
3. Dimensionality reduction & clustering:
   - PCA reduction (50 principal components)
   - UMAP reduction (2D visualization)
   - Leiden clustering (9 clusters: 0-8)
   - Generate cluster plot (`umap_clusters.png`)
4. Marker gene analysis:
   - Identify marker genes for each cluster
   - Generate heatmap (`marker_genes.png`)
5. Key gene analysis:
   - Analyze BECN1, FOXP3, CD4, CD8A, IL2RA, CTLA4
   - Generate UMAP plot (`key_genes_umap.png`)
   - Generate violin plot (`key_genes_violin.png`)
6. Treg cell identification:
   - Identify Treg cells based on high FOXP3 expression
   - Compare BECN1 expression in Treg vs. non-Treg cells
   - Generate Treg distribution plot (`treg_cells_umap.png`)
   - Generate BECN1-FOXP3 correlation plot (`becn1_foxp3_correlation.png`)

**Results:**

***Lung Dataset:***
1. T-cell subpopulation classification:
   - 9 identified subpopulations (0-8)
   - Clusters 6 and 8 show clear Treg characteristics (high FOXP3, IL2RA, CTLA4 expression)
2. BECN1-Treg relationship:
   - Terminal feedback: BECN1 expression significantly higher in Treg vs. non-Treg cells (0.277 vs 0.125, p=5.14e-16)
   - Higher BECN1 expression in clusters 6 and 8
   - Correlation between BECN1 and FOXP3 expression (evidenced by `becn1_foxp3_correlation.png`)

***Blood Dataset:***
- 10 clusters identified
- Potential Treg clusters: 8, 2, 9 (Cluster 2 shows stronger evidence from `key_genes_violin.png`)
- Treg cells: Average BECN1 expression = 0.236
- Non-Treg cells: Average BECN1 expression = 0.157
- T-test: t=6.37, p=1.87e-10

## 3. Subsequent Analysis: Precise Definition of CD4 Treg Cells

**Data Content:** Multiple marker genes (minimum 3)

1. `processed_data.h5ad`: Preprocessed data (use `readprocess_data.py` to view column names)
2. Use `distin.py` to add `is_CD4_Treg` marker column
3. Visualize CD4 Treg distribution

**Output Files:**
- `CD4_Treg_analyzed.h5ad` (data with CD4 Treg markers)
- `CD4_Treg_umap.png` (CD4 Treg distribution)
- `CD4_Treg_markers_umap.png` (marker gene expression)

## 4. Analysis of CD4 Treg Cells

Using `CD4_Treg_analyzed.h5ad`:
- Perform differential gene expression analysis on CD4 Treg cells
- Use `generelate.py` with gene name parameters (short commands)
- Output: `BECN1_expression_violin.png`

**Lung T-cell Dataset Final Interpretation:**
- BECN1 expression significantly higher in CD4 Treg vs. non-CD4 Treg cells (p<0.001)
- Moderate effect size (Cohen's d=0.43)
- Key finding: Expression frequency in CD4 Treg (39.17%) substantially higher than non-CD4 Treg (8.09%)
