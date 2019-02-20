import csv
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

# print("Performing PCA..")
# pca = PCA(n_components=2)
# pca_result = pca.fit_transform(df.values)
# df['pca-one'] = pca_result[:,0]
# df['pca-two'] = pca_result[:,1]
# print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

# x_size = 1000
# y_size = 1000
# num_genes, num_samples = pca_result.shape

# points = hv.Points(pca_result)
# histo = points.hist(num_bins=51, dimension=['x','y'])
# hv.save(histo,  'pca_'+ str(time.time()) + ".png")

# Create test sets
sex_labeled_samples = []
sex_labaled_full_samples = {}
sex_labels = {}

for title in metadata.keys():
	sample_metadata = metadata[title]
	if sample_metadata['refinebio_sex'] in ['male', 'female']:
		sex_labeled_samples.append(sample_metadata['refinebio_accession_code'])
		sex_labaled_full_samples[sample_metadata['refinebio_accession_code']] = sample_metadata
		# Label the sample in the frame - booleanize??
		try:
			df[sample_metadata['refinebio_accession_code']]['sex'] = sample_metadata['refinebio_sex']
		except Exception as e:
			print(e)

		if sample_metadata['refinebio_sex'] == 'female':
			sex_labels[sample_metadata['refinebio_accession_code']] = 0
		else:
			sex_labels[sample_metadata['refinebio_accession_code']] = 1


sex_labeled_samples = list(set(sex_labeled_samples) - set(['E-MEXP-405-HS04', 'E-MEXP-405-HS03', 'E-MEXP-405-HS02', 'E-MEXP-405-HS01', 'GSM1707858', 'GSM1707864', 'GSM1707855']))
sex_labeled_data = df[sex_labeled_samples]

sex_labeled_data = sex_labeled_data.transpose()

# Split into test and training data
df_train, df_test = train_test_split(sex_labeled_data, train_size=.7, test_size=.3, random_state=0, shuffle=True)

sex_labels_train = []
df_train_trans = df_train.transpose()
for col in df_train_trans.columns.to_list():
	sex_labels_train.append(sex_labels[col])

sex_labels_test = []
df_test_trans = df_test.transpose()
for col in df_test_trans.columns.to_list():
	sex_labels_test.append(sex_labels[col])

# KNN
knn = KNeighborsClassifier()
knn.fit(df_train, sex_labels_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(df_train, sex_labels_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(df_test, sex_labels_test)))

import pdb
pdb.set_trace()

# You have two options, bokeh requires selenium and phantomjs for png export, if you have those you can do 
# hv.save(hv_obj, 'test.png') 
# or you could use the matplotlib backend using 
# hv.save(hv_obj, 'test.png', backend='matplotlib')

dask_df = dd.from_pandas(df, npartitions=multiprocessing.cpu_count()).persist()
#dask_df = dd.read_csv('./data/DANIO_RERIO.tsv', sep='\t', header=0, index_col=0, error_bad_lines=False)
print("Converted to Dask..")

# des = datashade(hv.Points(dask_df))
# print("Datashaded..")

num_genes, num_samples = df.shape

print("To Dask Array..")
da = dask_df.to_dask_array(True).persist()

import pdb
pdb.set_trace()

print("To Image..")

x_size = 1000
y_size = 1000

img = hv.Image((np.arange(num_samples), np.arange(num_genes), da))
rasterized_img = rasterize(img, width=x_size, height=y_size)
rasterized_img.opts(width=x_size, height=y_size, cmap='viridis', logz=True)

import pdb
pdb.set_trace()

# You have two options, bokeh requires selenium and phantomjs for png export, if you have those you can do 
# hv.save(hv_obj, 'test.png') 
# or you could use the matplotlib backend using 
# hv.save(hv_obj, 'test.png', backend='matplotlib')

hv.save(rasterized_img, "rasterized" + str(time.time()) + ".png")
print("Wrote output..")


# heatmap = hv.HeatMap(dask_df, label='Zebrafish Samples')
# #aggregate = hv.Dataset(heatmap).aggregate('YEAR', np.mean, np.std)

# vline = hv.VLine(1963)
# marker = hv.Text(1964, 800, 'Vaccine introduction', halign='left')

# agg = hv.ErrorBars(aggregate) * hv.Curve(aggregate)

# overlay = (heatmap + agg * vline * marker).cols(1)
# overlay.opts(
#     opts.HeatMap(width=900, height=500, tools=['hover'], logz=True, 
#                    invert_yaxis=True, labelled=[], toolbar='above', xaxis=None),
#     opts.VLine(line_color='black'),
#     opts.Overlay(width=900, height=200, show_title=False, xrotation=90))

