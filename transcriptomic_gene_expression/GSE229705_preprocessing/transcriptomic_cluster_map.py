# load necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load training dataset - genes are the index
df = pd.read_csv("top20_DEGs.csv", index_col=0)
# remove version from ensembl IDs (everything after the first dot)
df.index = df.index.str.split('.').str[0]
# replace 'Tumor' with 'Tumour' in column names
df.columns = df.columns.str.replace("Tumor", "Tumour")

df.head()

# load gene names
genes = pd.read_csv("gene_names.csv", index_col=0)
genes = genes[genes.iloc[:, 0].notna()]  # remove NaNs
genes = genes[genes.iloc[:, 0].str.strip() != ""] # remove empty string values
genes

# create mapping dictionary: ensembl ID - gene 
id_to_gene = genes[genes.columns[0]].to_dict()

# map gene names, keeping original ensembl ID if no match found
df.index = df.index.map(lambda idx: id_to_gene.get(idx, idx))
df

# cluster map of samples and the top 20 DEGs

cpg = sns.clustermap(df,
                   cmap='viridis',
                   figsize=(16, 10),
                   cbar_kws={'label': 'VST Counts'},
                   metric='euclidean',
                   method='average',
                   yticklabels=True,
                   cbar_pos=[0.01, 0.82, 0.065, 0.2])

# adjust colorbar font size
cpg.ax_cbar.yaxis.label.set_size(22)  # colorbar label font size
cpg.ax_cbar.tick_params(labelsize=22) # colorbar tick font size

# add x and y axis labels with larger font
cpg.ax_heatmap.set_xlabel('Samples', fontsize=22, labelpad=15)
cpg.ax_heatmap.set_ylabel('DEGs', fontsize=22, labelpad=15)

# get current x tick labels
xticklabels = cpg.ax_heatmap.get_xticklabels()
# reduce x tick labels by spacing (better readability)
spacing = 2
reduced_labels = [label.get_text() for i, label in enumerate(xticklabels) if i % spacing == 0]
positions = [label.get_position()[0] for i, label in enumerate(xticklabels) if i % spacing == 0]
# apply reduced x ticks labels
cpg.ax_heatmap.set_xticks(positions)
cpg.ax_heatmap.set_xticklabels(reduced_labels, fontsize=20)
# make y ticks labels bigger
cpg.ax_heatmap.set_yticklabels(cpg.ax_heatmap.get_yticklabels(), fontsize=20)

plt.savefig("top20_DEGs_clustermap.png", bbox_inches='tight', dpi=300)
