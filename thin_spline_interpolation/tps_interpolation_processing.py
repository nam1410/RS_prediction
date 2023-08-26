import h5py
import pandas as pd
import numpy as np
import glob
import sys
import os
import heapq
from collections import Counter
all_blockmaps = glob.glob('<pathToFeatures>/h5_files/*.h5') #extract all the blockmap h5 files from your features directory mainly to grab the patch coordinates
all_csv = glob.glob('<pathToInterpolatedFeatures>/*_interpolated.csv')
def to_percentiles(scores):
    from scipy.stats import rankdata
    scores = rankdata(scores, 'average')/len(scores) * 100   #assigns rank to the data and deals with ties appropriately; calculates the percentage
    return scores
def compu_average(df_ann):
	combined_coords = list(zip(df_ann['coords_x'], df_ann['coords_y'])) # combine x and y coordinates
	coord_counts = Counter(combined_coords) # count occurrences of each coordinate
	duplicates = [coord for coord, count in coord_counts.items() if count > 1] # find duplicate coordinates
	duplicate_indices = [[i for i, coord in enumerate(combined_coords) if coord == dup] for dup in duplicates] # find indices of duplicates and corresponding z values
	z_values = [df_ann.iloc[index]['z'] for index in duplicate_indices]
	if len(duplicates) > 0:
		for f, b, i in zip(duplicates, duplicate_indices, z_values):
			avg = sum(i) / len(i)
			for x in range(len(b)):
				if x == 0:
					df_ann._set_value(b[x],'z',avg)
				else:
					df_ann._set_value(b[x],'z',np.nan)
		df_ann['coords_x'] = np.array(df_ann['coords_x'], dtype = int)
		df_ann['coords_y'] = np.array(df_ann['coords_y'], dtype = int)
		print('nan',df_ann.isnull().sum().sum())
		df_ann = df_ann.dropna()
	return df_ann
block_count = len(all_blockmaps)
block_copy = block_count
for i in all_blockmaps:
	slide_id = i.split(".")[<array_position>].split("/")[<array_position>].split("blockmap")[<array_position>] 
	csv_path = 'pathToInterpolatedFeatures/'+slide_id+'interpolated.csv'
	if csv_path in all_csv: 
		inter_path = '<pathToInterpolatedFeatures>'+slide_id+'interpolated_processed.csv'
		only_ann = '<pathToInterpolatedFeatures>'+slide_id+'interpolated_only_ann.csv'
		print('inter_path',inter_path)
		if not os.path.exists(only_ann):
			df_ann = pd.read_csv(csv_path)
			old_len = len(df_ann)
			df_att = h5py.File(i, "r")
			coords = np.array(df_att["coords"])
			scores = np.array(df_att["attention_scores"])
			print('length',len(scores))
			print(np.shape(scores))
			print('blockmap',i)
			print('csv_interpolation',csv_path)
			df_ann = compu_average(df_ann)
			coords_x, coords_y = np.split(coords,2,axis=1)
			if len(scores.shape)==2 and len(coords_x.shape) == 2 and len(coords_y.shape)==2:
				scores = scores.flatten() #flatten the array
				coords_x = coords_x.flatten()
				coords_y = coords_y.flatten()
			scores = to_percentiles(scores)
			df_block = pd.DataFrame(coords_x,columns=['coords_x'])
			df_block['coords_y'] = coords_y
			df_block['z']=scores
			ann_x, ann_y = np.array(df_ann['coords_x']),np.array(df_ann['coords_y'])
			if len(ann_x.shape) == 2 and len(ann_y.shape)==2:
				ann_x = ann_x.flatten()
				ann_y = ann_y.flatten()
			target_coords = np.vstack((ann_x, ann_y)).T.tolist()
			ann_z = np.array(df_ann['z'])
			cou = 0
			lis_only_ann = []
			for index, row in df_block.iterrows():
				if [row['coords_x'],row['coords_y']] in target_coords:
					ind = target_coords.index([row['coords_x'],row['coords_y']])
					lis_only_ann.append({'coords_x':row['coords_x'], 'coords_y':row['coords_y'], 'z': ann_z[ind]})
					df_block._set_value(index,'z',ann_z[ind])
					cou += 1
					
			print(cou)
			block_count -= 1
			print('\n #######remaining#####', block_count, ' of ', block_copy)
			df_only_ann = pd.DataFrame(lis_only_ann)
			df_only_ann.to_csv(only_ann, index = False)
			df_block.to_csv(inter_path, index=False)
			
		
