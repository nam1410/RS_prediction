import glob
import pandas as pd
import numpy as np
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import sys
path='<pathToInterpolatedFeatures>/*_fin.csv'
all_fins = glob.glob(path)
def tps_interpolate(x_new,y_new,slide_id):
	x_coords = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5]
	y_coords = [0, -1, -2, 1, 2, 0, 1, 2, -1, -2, 0, -1, -2, 1, 2]
	z_coords = [0, 0, 0, 0.4, 1, 1, 1,1, 0.6, 0, 0.5, 0.35, 0, 0.65, 1]
	'''
	Define a query grid
	'''
	xi = np.linspace(0, 1)
	yi = np.linspace(-2,2)
	xi, yi = np.meshgrid(xi, yi)
	'''
	Create a thin-plate spline interpolation function
	'''
	rbf = Rbf(x_coords, y_coords, z_coords, function='thin_plate')
	'''
	Evaluate the interpolation function at the query grid
	'''
	zi = rbf(xi, yi)
	'''
	Evaluate the interpolation function at the predefined data points
	'''
	z_coords_interp = rbf(x_coords, y_coords)
	'''
	Scale x_new from 0 to 1
	'''
	x_new = (x_new - 0) / (100 - 0)
	y_new = y_new.astype(int)
	z_new = rbf(x_new, y_new)
	return np.array(z_new)


for i in all_fins:
	print(i)
	df_fin = pd.read_csv(i)
	slide_id = i.split(".")[<array_position>].split("/")[<array_position>].split("fin")[<array_position>] 	slide_id = i.split(".")[<array_position>].split("/")[<array_position>].split("blockmap")[<array_position>] 
	x_new = np.array(df_fin['attention_scores'])
	y_new = np.array(df_fin['mode'])
	z_new = tps_interpolate(x_new,y_new,slide_id)
	z_new = z_new * 100
	z_new = np.clip(z_new,0,100)
	df_fin['z'] = z_new
	inter_path = '<pathToInterpolatedFeatures>/'+slide_id+'interpolated.csv'
	df_fin.to_csv(inter_path, index=False)
	
