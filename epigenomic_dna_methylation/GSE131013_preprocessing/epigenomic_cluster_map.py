# load necessary libraries
import pyreadr 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load the training data 
data = pyreadr.read_r("top20_bVals.rds") # read in rds file 
df = data[None] # methylation beta value matrix
df.columns = [col.replace("Tumor", "Tumour") for col in df.columns]
print(df.shape)
df.head()

# cluster map of samples and the top 20 DMCs

cpg = sns.clustermap(df,
                   cmap='viridis',
                   figsize=(16, 10),
                   cbar_kws={'label': 'Beta value'},
                   metric='euclidean',
                   method='average',
                   yticklabels=True,
                   cbar_pos=[0.01, 0.82, 0.065, 0.2])

# adjust colorbar font size
cpg.ax_cbar.yaxis.label.set_size(22)  
cpg.ax_cbar.tick_params(labelsize=22) 

# add x and y axis labels 
cpg.ax_heatmap.set_xlabel('Samples', fontsize=22, labelpad=15)
cpg.ax_heatmap.set_ylabel('DMCs', fontsize=22, labelpad=15)

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

# save plot as png
plt.savefig("top20_DMCs_clustermap.png", bbox_inches='tight', dpi=300)