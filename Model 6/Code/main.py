
import pandas as pd
import geopandas as gpd
import numpy as np
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns
import os

# To import the self made functions you need to fill in the path were you
# saved the functions instead of the word 'Code'
import sys
#sys.path.append('C:\\Users\\s158607\\Documents\\2019-2020\\Data challenge 3\\JBG060-Data-Challenge-3-master\\code')
sys.path.append('C:\\Users\\s158607\\PycharmProjects\\DataChallenge3\\Model 6\\code')

import utility
import load_files as lf
import flow_level_conversion as flc
import preprocessing as pre
import measurement_analysis as mea
import flow_model as model
import data_imputation as impu
import pred_to_rain as ptr

import warnings
warnings.filterwarnings('ignore')



PATH = 'C:\\Users\\s158607\\PycharmProjects\\DataChallenge3\\Model 6\\code\\' # CHANGE (!)
PATH_MEASUREMENTS = PATH + "sewer_data\\data_pump\\RG8150"
PATH_RAIN_DATA = PATH + "sewer_data\\rain_timeseries"
PATH_SHAPE_FILES = PATH + "sewer_model\\aa-en-maas_sewer_shp"
PATH_RAIN_PREDICTION = PATH + "sewer_data\\rain_grid_prediction"



# LOADING NECESSARY DATA
# Measurements
flow_data, level_data = lf.get_measurements(PATH_MEASUREMENTS)
flow_data = pre.fill_flow(pre.clean_mes_data(flow_data))
level_data = pre.fill_level(pre.clean_mes_data(level_data))

# # Actual rain
rain_data = lf.get_rain(PATH_RAIN_DATA)

# Rain predicton
pred_days, rain_prediction = lf.get_rain_prediction(PATH_RAIN_PREDICTION, reduce_grid = True)

# Shape file data frames
area_data = lf.sdf(PATH_SHAPE_FILES).area_data



# creating the dwaas haas table
area_data.crs = {'init': 'epsg:28992'}
measure = mea.measurement_analysis(flow_data, level_data, rain_data, area_data = area_data, village_code = 'DRU')
dwaas_table = measure.compare_flow()
dwaas_table



# creating the imputated flow
imputed_flow = impu.fill_flow(flow_data, level_data)
imputed_flow



# creating the prediction of the hourly flow
flow_model = model.flow_model(flow_data, level_data, (pred_days, rain_prediction))
cross_validation_score = flow_model.StochasticGradientDescent()
cross_validation_score

