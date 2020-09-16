Our code uses python version 3.7.3

The libaries that are used are:
	- pandas	0.24.2
	- geopandas	0.4.1
	- numpy		1.16.4
	- statsmodels	0.10.0
	- matplotlib	3.1.0
	- seaborn	0.9.0
	- sys
	- scipy		1.2.1
	- datetime
	- os
	- pickle
	- keras		2.2.4
	- tensorflow	1.9.0
	- sklearn 	0.21.2 (scikit-learn)	
	- holidays	0.9.11

In the file code you can find the core code of the project. Code consist of different .py files, which you can open with for
example pycharm.In code are the files:
	- utility: in here we got some important functions, like transforming the level of a round pipe to a normal level,
		finding the cummulative values, finding the next bigger/lower value, finding folders that are in the directory and
		finding folders in a directory
	- load_files: In this file the level, flow, rain prediction, shp files and actual rain data is opened. After this it is stored
		in a SQL database
	- preprocessing: In this file the data has been cleaned, the missing data is filled in, there are groups created for flow and
		level, rain data is summarized, the hourly flow is calculated and the predicted rain is matched with the hourly flow using
		the timestamp.
	- measurement_analysis: Here we create the same results as in the dwaas haas analysis
	- pred_to_rain: Find the cell in the rain prediction grid corresponding to the centroid of an area.
	- flow_level_conversion: Estimates a coefficient between total flow through a pump and level change. With this, sewer
		capacity and sewer intake over a period of time can be estimated.
	- data_imputation: Estimates missing measurements of flow based on the level and slope of level at this time.
	- flow_model: Here a model is build to predict the hourly flow for the next 24 hours

In the file running the code you can find two jupiter notebook files:
	- notebook inflow analysis: which consist of creating an inflow coefficient
	- run code: which runs the most important functions from the utility file.
In these jupyter notebooks the file path needs to be changed.
