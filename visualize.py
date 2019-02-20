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

hv.extension('bokeh')

print("Loading data..")
df = pd.read_csv('./data/DANIO_RERIO.tsv', sep='\t', header=0, index_col=0, error_bad_lines=False)
print("Data loaded..")

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

