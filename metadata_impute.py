import csv
import json
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

refinebio_metadata_fields = [
	# 'refinebio_accession_code',
	# 'refinebio_age',
	# 'refinebio_cell_line',
	# 'refinebio_compound',
	# 'refinebio_disease',
	# 'refinebio_disease_stage',
	# 'refinebio_genetic_information',
	# 'refinebio_organism',
	# 'refinebio_platform',
	# 'refinebio_race',
	'refinebio_sex',
	# 'refinebio_source_archive_url',
	# 'refinebio_source_database',
	'refinebio_specimen_part',
	# 'refinebio_subject',
	# 'refinebio_time',
	# 'refinebio_title',
	# 'refinebio_treatment',
]

num_imputed_fields = 0
imputed_metadata = metadata.copy()

for refinebio_field in refinebio_metadata_fields:
	print("\n\nClustering and imputing metadata field: " + refinebio_field)

	metadata_labeled_samples = []
	metadata_labaled_full_samples = {}
	metadata_labels = {}
	ordmap = {}

	for title in metadata.keys():
		sample_metadata = metadata[title]
		if sample_metadata[refinebio_field] not in [None, '']:
			metadata_labeled_samples.append(sample_metadata['refinebio_accession_code'])
			metadata_labaled_full_samples[sample_metadata['refinebio_accession_code']] = sample_metadata

			# Put the result in the data itself?
			# try:
			# 	df[sample_metadata['refinebio_accession_code']][refinebio_field] = sample_metadata[refinebio_field]
			# except Exception as e:
			# 	print(e)

			# XXX: Does this need to be booleanized?
			# XXX: This is a hack! Don't do this!
			ordsum = sum([ord(x) for x in sample_metadata[refinebio_field]])
			if ordsum not in ordmap.keys():
				ordmap[ordsum] = sample_metadata[refinebio_field]
			metadata_labels[sample_metadata['refinebio_accession_code']] = ordsum


	print("Mapped strings to integers: ")
	print(ordmap)

	if ordmap == {}:
		continue

	# We have to do this in a stupid way because of frames in the data not matching they're accession codes
	metadata_labeled_data = pd.DataFrame()
	frames_to_merge = []
	for sample in metadata_labeled_samples:
		try:
			sample_frame = df[sample]
			frames_to_merge.append(sample_frame)
		except Exception as e:
			pass
			# #print(e)
			# print("Skipping poorly labelled sample " + str(sample))

	metadata_labeled_data = pd.concat(frames_to_merge, axis=1, keys=None, join='outer', copy=False, sort=True)
	metadata_labeled_data = metadata_labeled_data.transpose()

	# Split into test and training data
	df_train, df_test = train_test_split(metadata_labeled_data, train_size=.7, test_size=.3, random_state=0, shuffle=True)

	metadata_labels_train = []
	df_train_trans = df_train.transpose()
	for col in df_train_trans.columns.to_list():
		metadata_labels_train.append(metadata_labels[col])

	metadata_labels_test = []
	df_test_trans = df_test.transpose()
	for col in df_test_trans.columns.to_list():
		metadata_labels_test.append(metadata_labels[col])

	# KNN
	print('')
	knn = KNeighborsClassifier()
	knn.fit(df_train, metadata_labels_train)
	num_cats = len(set(metadata_labels_train))
	print('Accuracy of kNN classifier on training set (' + str(len(df_train)) + ' labeled samples, ' + str(num_cats) + ' total categories) for metadata field ' + refinebio_field + ': {:.2f}'
	     .format(knn.score(df_train, metadata_labels_train)))
	print('Accuracy of kNN classifier on test set (' + str(len(df_test)) + ' labeled samples, ' + str(num_cats) + ' total categories) for metadata field ' + refinebio_field + ': {:.2f}'
	     .format(knn.score(df_test, metadata_labels_test)))

	for title in metadata.keys():
		sample_metadata = metadata[title]
		if sample_metadata[refinebio_field] in [None, '']:

			# Get the frame
			try:
				sample_frame = df[sample_metadata['refinebio_accession_code']]
			except Exception as e:
				print ("No frame!: " + sample_metadata['refinebio_accession_code'])
				try:
					sample_frame = df[sample_metadata['refinebio_title']]
				except Exception as e:
					print ("No title!: " + sample_metadata['refinebio_title'])
					continue

			# Classify it
			prediction = knn.predict([sample_frame])
			mapped_prediction = ordmap[prediction[0]]
			print(ordmap[prediction[0]])

			probabilities = knn.predict_proba([sample_frame])
			probability = sorted(probabilities[0])[-1]

			# Put it in our imputed metadata object
			imputed_metadata[title][refinebio_field] = mapped_prediction
			print("Imputed value " + str(mapped_prediction) + " for value " + refinebio_field + " in sample " + title + " with probability " + str(probability) + ".")

			num_imputed_fields = num_imputed_fields + 1

# Save our imputed object
with open('metadata_with_imputation.json', 'w') as outfile: 
	json.dump(imputed_metadata, outfile, sort_keys=True, indent=4, separators=(',', ': '))
