import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sns

df = pd.read_csv('./data/DANIO_RERIO.tsv', sep='\t', header=0, index_col=0, error_bad_lines=False)
df.head()

# heatmap = sns.heatmap(df)
# heatmap.savefig("heatmap_" + str(time.time()) + ".png")

import pdb
pdb.set_trace()


clustermap = sns.clustermap(df)
clustermap.savefig("clustermap_" + str(time.time()) + ".png")