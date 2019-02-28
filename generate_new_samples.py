import csv
import json
import random
import time

import datashader as ds
import pandas as pd
import numpy as np
import holoviews as hv
from holoviews import opts
from colorcet import fire
from datashader import transfer_functions as tf
from dask import dataframe as dd
import multiprocessing

from holoviews.operation.datashader import datashade, shade, dynspread, rasterize
from holoviews.operation import decimate
from holoviews import opts

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from fancyimpute import KNN, BiScaler, SoftImpute, IterativeSVD

hv.extension('bokeh')

print("Loading data..")
df = pd.read_csv('./data/DANIO_RERIO.tsv', sep='\t', header=0, index_col=0, error_bad_lines=False)

print("Loading metadata..")
metadata = {}
with open('./data/metadata_DANIO_RERIO.tsv', 'r') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
    	metadata[row['refinebio_title']] = row

print("All data loaded..")

# Choose a sample of desired type
# Copy it
# Drop n% of non-imputed values
	# Impute
# Repeat

# Choose a sample. This should ideally be based on a property
sample_copy = df[df.columns[0]]
df['SYNTHETIC'] = sample_copy

# Set and impute the values, n% at a time
iteration_percent = .20
all_rows = df.index
all_cols = df.columns
rows_to_impute = set(df.index)
imputed_rows = set()

while len(rows_to_impute) > 0:
	try:
		impute_me = set(random.sample(rows_to_impute, int(len(all_rows) * iteration_percent)))
	except Exception:
		# Population larger than sample
		impute_me = rows_to_impute
	rows_to_impute = rows_to_impute - impute_me

	df['SYNTHETIC'][impute_me] = np.nan

	needs_imputation_transposed = df.transpose()
	print("Imputing step!")
	imputed_matrix = IterativeSVD(rank=10).fit_transform(needs_imputation_transposed)
	imputed_matrix_transposed = imputed_matrix.transpose()
	print("Imputed!")

	# Convert back to Pandas
	df = df.transpose()
	df_imputed_matrix_transposed = pd.DataFrame.from_records(imputed_matrix_transposed)
	df_imputed_matrix_transposed.index = all_rows
	df_imputed_matrix_transposed.columns = all_cols
	df = df_imputed_matrix_transposed

import pdb
pdb.set_trace()

df.to_csv('synthetic.tsv', sep='\t', encoding='utf-8')
