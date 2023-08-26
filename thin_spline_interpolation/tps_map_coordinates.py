import h5py
import pandas as pd
import numpy as np
import glob
import sys
import os
import heapq
all_blockmaps = glob.glob('<pathToFeatures>/h5_files/*.h5') #extract all the blockmap h5 files from your features directory mainly to grab the patch coordinates
all_csv = glob.glob('<pathToAnnotations>/*.csv')
count = 0
matches = 0
true_number = 0
def to_percentiles(scores):
    from scipy.stats import rankdata
    scores = rankdata(scores, 'average')/len(scores) * 100   #assigns rank to the data and deals with ties appropriately; calculates the percentage
    return scores
def find_nearest_left_top(ann_x_num, ann_y_num, att_x, att_y):
	global matches
	lesser_x = min(filter(lambda x: x <= ann_x_num, att_x), key=lambda x: abs(x - ann_x_num), default="EMPTY")
	lesser_y = min(filter(lambda x: x <= ann_y_num, att_y), key=lambda x: abs(x - ann_y_num), default="EMPTY")
	lesser_x_all_indices = [i for i, x in enumerate(att_x) if x == lesser_x]
	lesser_y_all_indices = [i for i, x in enumerate(att_y) if x == lesser_y]
	match = set(lesser_x_all_indices) & set(lesser_y_all_indices)
	if len(match) != 0:
		matches += 1
		return match
	else:
		greater_x = min(filter(lambda x: x >= ann_x_num, att_x), key=lambda x: abs(x - ann_x_num),default="EMPTY")
		lesser_y = min(filter(lambda x: x <= ann_y_num, att_y), key=lambda x: abs(x - ann_y_num), default="EMPTY")
		greater_x_all_indices = [i for i, x in enumerate(att_x) if x == greater_x]
		lesser_y_all_indices = [i for i, x in enumerate(att_y) if x == lesser_y]
		match = set(greater_x_all_indices) & set(lesser_y_all_indices)
		if len(match) != 0:
			matches += 1
			return match
		else:
			lesser_x = min(filter(lambda x: x <= ann_x_num, att_x), key=lambda x: abs(x - ann_x_num), default="EMPTY")
			greater_y = min(filter(lambda x: x >= ann_y_num, att_y), key=lambda x: abs(x - ann_y_num), default="EMPTY")
			lesser_x_all_indices = [i for i, x in enumerate(att_x) if x == lesser_x]
			greater_y_all_indices = [i for i, x in enumerate(att_y) if x == greater_y]
			match = set(lesser_x_all_indices) & set(greater_y_all_indices)
			if len(match) != 0:
				matches += 1
				return match
			else:
				greater_x = min(filter(lambda x: x >= ann_x_num, att_x), key=lambda x: abs(x - ann_x_num), default="EMPTY")
				greater_y = min(filter(lambda x: x >= ann_y_num, att_y), key=lambda x: abs(x - ann_y_num), default="EMPTY")
				greater_x_all_indices = [i for i, x in enumerate(att_x) if x == greater_x]
				greater_y_all_indices = [i for i, x in enumerate(att_y) if x == greater_y]
				match = set(greater_x_all_indices) & set(greater_y_all_indices)
				if len(match) != 0:
					matches += 1
					return match
				else:
					min_abs_x = min(att_x, key=lambda x:abs(x-ann_x_num), default="EMPTY")
					min_abs_y = min(att_y, key=lambda x:abs(x-ann_y_num), default="EMPTY")
					x_diff = np.abs(np.array(att_x) - ann_x_num)
					indices_x = np.argsort(x_diff)[:1000]
					y_diff = np.abs(np.array(att_y) - ann_y_num)
					indices_y = np.argsort(y_diff)[:1000]
					new_match = set(indices_x) & set(indices_y)
					if len(new_match) != 0:
						new_match_elements = [(att_x[i],att_y[i],i) for i in new_match]
						reference = [ann_x_num,ann_y_num]
						nearest = min(new_match_elements, key=lambda x: ((x[0] - reference[0])**2 + (x[1] - reference[1])**2)**0.5, default="EMPTY")
						matches += 1
						return nearest[2]


def linear_search(top_x, top_y, bottom_x, bottom_y, coords_x, coords_y):
	combined_coords = np.vstack((coords_x, coords_y)).T.tolist()
	seq = 0
	coo = 0
	coords_scores = []
	ref = None
	for [x,y] in combined_coords:
		if (x < bottom_x+1) and (x >= top_x) and (y < bottom_y+1) and (y >= top_y):
			ref = [x,y]
			print(ref)
			break
	if ref is None:
		print(top_x, top_y, bottom_x, bottom_y, coords_x, coords_y)
		sys.exit()
	return ref
	
def compute_area(box_coordinates, coords_x, coords_y, scores):
	global true_number
	coords_scores = []
	combined_coords = np.vstack((coords_x, coords_y)).T.tolist()
	counter_neg = len(box_coordinates) 
	counter_neg_copy = counter_neg
	print('len', counter_neg)
	for i in box_coordinates:
		processed = False
		print(i['top_left_match'],i['bottom_right_match'],i['top_right_match'], i['bottom_left_match'], i['reference_point'], processed)
		
		if (i['top_left_match'] != None) and (i['bottom_right_match'] != None) and (processed == False):
			if isinstance(i['top_left_match'], set):
				top_left_index = i['top_left_match'].pop()
			else:
				top_left_index = i['top_left_match']
			top_left_x, top_left_y = combined_coords[top_left_index]
			if isinstance(i['bottom_right_match'], set):
				bottom_right_index = i['bottom_right_match'].pop()
			else:
				bottom_right_index = i['bottom_right_match']
			bottom_right_x, bottom_right_y = combined_coords[bottom_right_index]
			for x in range(top_left_x,bottom_right_x+1, 256):
				for y in range(top_left_y, bottom_right_y+1, 256):
					if [x,y] in combined_coords:
						if i['flip'] == 1:
							flip_score = 100 - scores[combined_coords.index([x,y])]
						else:
							flip_score = scores[combined_coords.index([x,y])]
						coords_scores.append([x,y,scores[combined_coords.index([x,y])], i['flip'], flip_score,i['mode']])
						true_number += 1
			processed = True

			
		if (i['top_right_match'] != None) and (i['bottom_left_match'] != None) and (processed == False):
			if isinstance(i['top_right_match'], set):
				top_right_index = i['top_right_match'].pop()
			else:
				top_right_index = i['top_right_match']
			top_right_x, top_right_y = combined_coords[top_right_index]
			if isinstance(i['bottom_left_match'], set):
				bottom_left_index = i['bottom_left_match'].pop()
			else:
				bottom_left_index = i['bottom_left_match']
			bottom_left_x, bottom_left_y = combined_coords[bottom_left_index]
			for x in range(top_right_x,bottom_left_x, -256):
				for y in range(top_right_y, bottom_left_y, 256):	
					if [x,y] in combined_coords:
						if i['flip'] == 1:
							flip_score = 100 - scores[combined_coords.index([x,y])]
						else:
							flip_score = scores[combined_coords.index([x,y])]
						coords_scores.append([x,y,scores[combined_coords.index([x,y])], i['flip'], flip_score,i['mode']])
						true_number += 1
			processed = True	
		if (i['reference_point'] != None) and (processed == False):
			for x in range(int(i['reference_point'][0]),int(i['bottom_right_coords'][0])+1, 256):
				for y in range(int(i['reference_point'][1]), int(i['bottom_right_coords'][1])+1, 256):
					if [x,y] in combined_coords:
						if i['flip'] == 1:
							flip_score = 100 - scores[combined_coords.index([x,y])]
						else:
							flip_score = scores[combined_coords.index([x,y])]
						coords_scores.append([x,y,scores[combined_coords.index([x,y])], i['flip'], flip_score,i['mode']])
						true_number += 1
				for y in range(int(i['reference_point'][1]), int(i['top_right_coords'][1]), -256):
					if [x,y] in combined_coords:
						if i['flip'] == 1:
							flip_score = 100 - scores[combined_coords.index([x,y])]
						else:
							flip_score = scores[combined_coords.index([x,y])]
						coords_scores.append([x,y,scores[combined_coords.index([x,y])], i['flip'], flip_score,i['mode']])
						true_number += 1
			for x in range(int(i['reference_point'][0]), int(i['top_left_coords'][0]), -256):
				for y in range(int(i['reference_point'][1]), int(i['top_left_coords'][1]), -256):
					if [x,y] in combined_coords:
						if i['flip'] == 1:
							flip_score = 100 - scores[combined_coords.index([x,y])]
						else:
							flip_score = scores[combined_coords.index([x,y])]
						coords_scores.append([x,y,scores[combined_coords.index([x,y])], i['flip'], flip_score,i['mode']])
						true_number += 1
				for y in range(int(i['reference_point'][1]), int(i['bottom_left_coords'][1]), 256):
					if [x,y] in combined_coords:
						if i['flip'] == 1:
							flip_score = 100 - scores[combined_coords.index([x,y])]
						else:
							flip_score = scores[combined_coords.index([x,y])]
						coords_scores.append([x,y,scores[combined_coords.index([x,y])], i['flip'], flip_score,i['mode']])
						true_number += 1
			processed = True
		counter_neg_copy -= 1
		print('remaining ',counter_neg_copy, ' of ',counter_neg, ' processed ', processed)					
	return coords_scores	


top_bottom = 0
block_count = 0
for i in all_blockmaps:
	slide_id = i.split(".")[<array_position>].split("/")[<array_position>].split("blockmap")[<array_position>] 
	csv_path = '<pathToAnnotations>/'+slide_id+'annotation.csv'
	if csv_path in all_csv: 
		fin_path = '<pathToInterpolatedFeatures>/'+slide_id+'fin.csv'
		if not os.path.exists(fin_path):
			df_ann = pd.read_csv(csv_path)
			df_att = h5py.File(i, "r")
			coords = np.array(df_att["coords"])
			scores = np.array(df_att["attention_scores"])
			coords_x, coords_y = np.split(coords,2,axis=1)
			if len(scores.shape)==2 and len(coords_x.shape) == 2 and len(coords_y.shape)==2:
				scores = scores.flatten() #flatten the array
				coords_x = coords_x.flatten()
				coords_y = coords_y.flatten()
			scores = to_percentiles(scores)
			box_coordinates = []
			ret_no_points = None
			block_count += 1
			print('block_count',block_count)
			for index, row in df_ann.iterrows():
				top_left = [int(row['x']),int(row['y'])]
				bottom_right = [int(row['x']) + int(row['width']) ,int(row['y']) + int(row['height'])]
				top_right = [int(row['x']) + int(row['width']) ,int(row['y'])]
				bottom_left = [int(row['x']), int(row['y']) + int(row['height'])]
				top_left_match = find_nearest_left_top(top_left[0], top_left[1], coords_x, coords_y)
				bottom_right_match = find_nearest_left_top(bottom_right[0], bottom_right[1], coords_x, coords_y)
				top_right_match = find_nearest_left_top(top_right[0], top_right[1], coords_x, coords_y)
				bottom_left_match = find_nearest_left_top(bottom_left[0], bottom_left[1], coords_x, coords_y)
				if (top_left_match != None) and (bottom_right_match != None):
					top_bottom += 1
				if (top_right_match != None) and (bottom_left_match != None):
					top_bottom += 1
				else:  
					ret_no_points = linear_search(top_left[0], top_left[1], bottom_right[0], bottom_right[1], coords_x, coords_y)
				box_coordinates.append({'top_left_coords':[top_left[0], top_left[1]],'top_left_match':top_left_match, 'bottom_right_coords':[bottom_right[0], bottom_right[1]], 'bottom_right_match':bottom_right_match, 'top_right_coords':[top_right[0], top_right[1]],'top_right_match':top_right_match, 'bottom_left_coords':[bottom_left[0], bottom_left[1]], 'bottom_left_match':bottom_left_match, 'reference_point':ret_no_points, 'mode': row['mode'], 'flip':row['flip']})
				ret_no_points = None
			coor = compute_area(box_coordinates, coords_x, coords_y, scores)
			df_feed_path = pd.DataFrame(coor, columns = ['coords_x', 'coords_y','old_attention_scores', 'flip', 'attention_scores', 'mode'])
			df_feed_path.to_csv(fin_path, index=False)
print("matches",matches)
print('top_bottom', top_bottom)
print("true_number",true_number)
