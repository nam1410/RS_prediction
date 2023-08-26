import h5py
import pandas as pd
import numpy as np
import glob
import sys
import os
import heapq
from collections import Counter
all_blockmaps = glob.glob('<pathToFeatures>/h5_files/*.h5') #extract all the blockmap h5 files from your features directory mainly to grab the patch coordinates
all_csv = glob.glob('<pathToInterpolatedFeatures>/*_interpolated_processed.csv')
for i in all_blockmaps:
	slide_id = i.split(".")[<array_position>].split("/")[<aray_position>] #extract the slide id
	csv_path = '<pathToInterpolatedFeatures>/'+slide_id+'_interpolated_processed.csv'
	if csv_path in all_csv: 
		df_att = h5py.File(i, "r")
		coordinates = np.array(df_att["coords"])
		coords_x, coords_y = np.split(coordinates, 2, axis =1)
		if len(coords_x.shape) == 2 and len(coords_y.shape) == 2:
			coords_x = coords_x.flatten()
			coords_y = coords_y.flatten()
		coordin = np.vstack((coords_x,coords_y)).T.tolist()
		df_block = pd.DataFrame(coords_x,columns=['coords_x'])
		df_block['coords_y'] = coords_y
		df_ann = pd.read_csv(csv_path)
		ann_x, ann_y = np.array(df_ann['coords_x']),np.array(df_ann['coords_y'])
		if len(ann_x.shape) == 2 and len(ann_y.shape)==2 :
			ann_x = ann_x.flatten()
			ann_y = ann_y.flatten()
		target_coords = np.vstack((ann_x, ann_y)).T.tolist()
		z_new = []
		for i in coordin:
			if i in target_coords:
				ind = target_coords.index(i)
				z_new.append(df_ann.iloc[ind]['z'])
			else:
				z_new.append(0)
		print('z_new',len(z_new))
		print('target_coords', len(target_coords))
		print('coordin',len(coordin))
		df_block['z'] = np.array(z_new)
		inter_path = '<pathToInterpolatedFeatures>/'+slide_id+'_interpolated_processed_coordinates.csv'
		print(inter_path)
		df_block.to_csv(inter_path, index=False)
		
