# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Loads data as provided by Aa-en-Maas in various ways.   #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

import pandas as pd
import numpy as np
import datetime
import geopandas as gpd
import os
import pickle
import utility


# Codes of pumps
pump_to_id_dict = {"Drunen": 8150,
                   "Haarsteeg": 8170,
                   "Oude Engelenseweg": 401,
                   "Helftheuvelweg": 301,
                   "Engelerschans": 201,
                   "De Rompert": 501,
                   "Maaspoort": 501}


def get_measurements(path, convert_time=True):
    """
    Will read all measurement data from given path and store them in separate dataframes.
    ~~~ EXAMPLE CALL ~~~
    flow_data, level_data = get_measurements("C:/mypath/RG8150")
    ~~~~~~~~~~~~~~~~~~~~
    """
    files = os.listdir(path)
    
    data = [pd.read_csv(path + "/" + i, sep = ";") for i in files]
    data =  pd.concat(data, sort = False, ignore_index = True)
    
    data["RG_ID"] = data["Tagname"].str.slice(9,13).astype(int)
    data["Value"] = data["Value"].str.replace(",", ".").astype(float)
    data["DataQuality"] = (data["DataQuality"] == "Good").astype(int)
    if convert_time == True:
        data["TimeStamp"] = pd.to_datetime(data["TimeStamp"], format="%d-%m-%Y %H:%M:%S")
        
    data = data[["Tagname", "RG_ID", "TimeStamp", "Value", "DataQuality"]]
    
    flow_data = data[data["Tagname"].str.contains("Debietmeting")].reset_index(drop = True)
    level_data = data[data["Tagname"].str.contains("Niveaumeting")].reset_index(drop = True)
    
    flow_data.drop("Tagname", axis=1, inplace=True)
    level_data.drop("Tagname", axis=1, inplace=True)
    
    return flow_data, level_data


def load_all_pumps(path, convert_time=True):
    """
    Will read all measurement data from given path and store them in separate dataframes.
    The format of all data sources is standardized.
    ~~~ EXAMPLE CALL ~~~
    level_bokhoven = load_all_pumps(path+"Data 1/sewer_data/data_pump/RG8180_L0")
    ~~~~~~~~~~~~~~~~~~~~
    """
    files = os.listdir(path)
    
    # OLD TYPE FLOW AND LEVEL VALUES FOR HAARSTEEG AND DRUNEN
    if ("RG8150" in path) or ("RG8170" in path):
        return get_measurements(path, convert_time=convert_time)
        
    
    # NEW TYPE FLOW AND LEVEL VALUES FOR HAARSTEEG AND BOKHOVEN
    if ("RG8180_L0" in path) or ("RG8180_Q0" in path) or ("rg8170_N99" in path) or ("rg8170_99" in path):
        data = [pd.read_csv(path + "/" + i, sep = ",") for i in files if ".csv" in i]
        data =  pd.concat(data, sort = False, ignore_index = True)
        
        data["RG_ID"] = data["historianTagnummer"].str.slice(9,13).astype(int)
        data["Value"] = data["hstWaarde"]
        
        data["DataQuality"] = (data["historianKwaliteit"] == 100).astype(int)
        
        if convert_time == True:
            data["datumBeginMeting"] = pd.to_datetime(data["datumBeginMeting"]).dt.strftime("%d-%m-%Y %H:%M:%S")
        
        data.rename(columns={"datumBeginMeting": "TimeStamp"}, inplace=True)
        
        return data[["RG_ID", "TimeStamp", "Value", "DataQuality"]]
    
    
    # LEVEL VALUES OF SMALL PUMPS
    if "data_pump_level" in path:
        data = [pd.read_csv(path + "/" + i, sep = ";") for i in files if ".csv" in i]
        data =  pd.concat(data, sort = False, ignore_index = True)
        
        data.rename(columns={"002: Oude Engelenseweg Niveau actueel (1&2)(cm)": "Oude Engelenseweg",
                             "003: Helftheuvelweg Niveau (cm)": "Helftheuvelweg",
                             "004: Engelerschans Niveau trend niveau DWA(cm)": "Engelerschans",
                             "005: De Rompert Niveau (cm)": "De Rompert",
                             "006: Maaspoort Niveau actueel (1&2)(cm)": "Maaspoort"}, inplace=True)

        data['TimeStamp'] = data['Datum'] + " " + data['Tijd']
        
        if convert_time == True:
            data["TimeStamp"] = pd.to_datetime(data["TimeStamp"], format="%d-%m-%Y %H:%M:%S")

        for i in ["Oude Engelenseweg", "Helftheuvelweg", "Engelerschans", "De Rompert", "Maaspoort"]:
            data.loc[:, i] = data.loc[:, i].str.replace(",", ".").astype(float)
        
        data_len = len(data)
        data = pd.concat([data[["TimeStamp", "Oude Engelenseweg"]].rename(columns={"Oude Engelenseweg": "Value"}),
                           data[["TimeStamp", "Helftheuvelweg"]].rename(columns={"Helftheuvelweg": "Value"}),
                           data[["TimeStamp", "Engelerschans"]].rename(columns={"Engelerschans": "Value"}),
                           data[["TimeStamp", "De Rompert"]].rename(columns={"De Rompert": "Value"}),
                           data[["TimeStamp", "Maaspoort"]].rename(columns={"Maaspoort": "Value"})],
                          axis=0, ignore_index=True)

        data["RG_ID"] = list(map(lambda i: pump_to_id_dict[i],
                                 np.repeat(["Oude Engelenseweg", "Helftheuvelweg",
                                            "Engelerschans", "De Rompert", "Maaspoort"], data_len)))
        
        return data
    

    # NEW TYPE FLOW OF WWTP AND SMALL PUMPS
    if ("data_pump_flow" in path) or ("data_wwtp_flow" in path):
        data = [pd.read_csv(path + "/" + i, sep = ",") for i in files if ".csv" in i]
        data =  pd.concat(data, sort = False, ignore_index = True)

        if "data_pump_flow" in path:
            data["RG_ID"] = data["historianTagnummer"].str.slice(26,29).astype(int)
        else:
            if "1882" in path:
                data["RG_ID"] = 1882
            elif "1876" in path:
                data["RG_ID"] = 1876
            else:
                data["RG_ID"] = 0
            
        data["Value"] = data["hstWaarde"]
        data["DataQuality"] = (data["historianKwaliteit"] == 100).astype(int)

        data["TimeStamp"] = pd.to_datetime(data["datumBeginMeting"]).dt.strftime('%d-%m-%Y %H:%M:%S')
        data = data[["RG_ID", "TimeStamp", "Value", "DataQuality"]]
        
        return data


def get_rain_prediction(path, from_date=None, to_date=None, reduce_grid=False):
    """
    Will read rain prediction data + dates from file names from given path and store those
    in separate dataframes.
    ~~~ EXAMPLE CALL ~~~
    pred_dates, pred_data = get_rain_prediction("C:/mypath/knmi....")
    ~~~~~~~~~~~~~~~~~~~~
    
    reduce_grid :    Skims down the data to the relevant area. Highly recommended if
                     your PC runs <16GB RAM.
    """
    files = os.listdir(path)
    
    dates = pd.Series(pd.to_datetime([i.split("_")[3] for i in files if ".aux" not in i]))
    
    if (from_date is not None) & (to_date is not None):
        boolean_ = (dates >= pd.to_datetime(from_date)) & (dates < pd.to_datetime(to_date))
        files = pd.Series(files)[boolean_]
    
    pred_date = pd.Series(pd.to_datetime([i.split("_")[2] for i in files if ".aux" not in i]))
    start_date = pd.Series(pd.to_datetime([i.split("_")[3] for i in files if ".aux" not in i]))
    end_date = pd.Series(pd.to_datetime([i.split("_")[4][:20] for i in files if ".aux" not in i]))
    
    if reduce_grid:                                            #Y: 51.830-51.321 X: 5.068-6.048
        data = np.array([np.loadtxt(path + "/" + i, skiprows=7)[91:(195+1), 101:(223+1)]
                         for i in files if ".aux" not in i])
    else:
        data = np.array([np.loadtxt(path + "/" + i, skiprows=7) for i in files if ".aux" not in i])
    
    date_data = pd.concat([pred_date, start_date, end_date], axis=1)
    date_data.columns = ["pred", "start", "end"]
    
    return date_data, data


def get_rain(path, convert_time=True):
    """
    Will read all rain data from given path and store them in a single dataframe.
    ~~~ EXAMPLE CALL ~~~
    rain_data = get_rain("C:/mypath/rain_timeseries")
    ~~~~~~~~~~~~~~~~~~~~
    """
    
    files = os.listdir(path)
    
    data = [pd.read_csv(path + "/" + i, skiprows=2) for i in files]
    data =  pd.concat(data, sort = False, ignore_index = True)
    if convert_time == True:
        data["Begin"] = pd.to_datetime(data["Begin"], format="%d-%m-%Y %H:%M:%S")
        data["Eind"] = pd.to_datetime(data["Eind"], format="%d-%m-%Y %H:%M:%S")
    
    data.rename({"Begin": "Start", "Eind": "End"}, axis=1, inplace = True)
    
    return data


def get_system_register(path):
    """
    Will read system information data from given path and store it in a single dataframe.
    ~~~ EXAMPLE CALL ~~~
    system_data = get_system_register("C:/mypath/sewer_model")
    ~~~~~~~~~~~~~~~~~~~~
    (!) This is almost the same data as the Rioleringsdeelgebied.shp file provides and it
    is recommended to use this one instead, see the class sdf.
    """
    
    system_data = pd.read_excel(path + "/" + "20180717_dump riodat rioleringsdeelgebieden.xlsx", skiprows=9)
    system_data = system_data[["Volgnr", "Code", "Naam / lokatie", "RWZI"]]
    system_data.columns = ["area_ID", "sewer_system", "area_name", "RWZI"]
    
    return system_data


class sdf:
    """
    Will read all shp files from given path and store them within this class as data frames.
    ~~~ EXAMPLE CALL ~~~
    data = sdf("C:/mypath/aa-en-maas_sewer_shp")
    data.area_data
    ~~~~~~~~~~~~~~~~~~~~
    """
    def __init__(self, path):
        # Sewage area data
        area_data = gpd.read_file(path + "/" + "Rioleringsdeelgebied.shp")
        area_data["area"] = area_data.area
        area_data = area_data[["RGDIDENT", "NAAMRGD", "RGDID", "area", "geometry"]]
        area_data.columns = ["sewer_system", "area_name", "area_ID", "area", "geometry"]
        
        # RG data
        RG_data = gpd.read_file(path + "/" + "Rioolgemaal.shp")
        RG_data = RG_data[["ZRE_ID", "ZREIDENT", "ZRW_ZRW_ID", "ZRGCAPA1",
                               "ZRE_ZRE_ID", "ZRGRGCAP",
                               "ZRGGANGL", "geometry"]]
        RG_data.columns = ["unit_ID", "RG_ID", "RWZI_ID", "min_capacity", "to_unit_ID", "max_capacity",
                           "RG_name", "geometry"]
        
        # RWZI regions
        RWZI_regions = gpd.read_file(path + "/" + "Zuiveringsregio.shp")
        RWZI_regions = RWZI_regions[["GAGNAAM", "geometry"]]
        RWZI_regions.columns = ["RWZI_name", "geometry"]
        
        # RWZI data
        RWZI_data = gpd.read_file(path + "/" + "RWZI.shp")
        RWZI_data = RWZI_data[["ZRW_ID", "ZRWIDENT", "ZRWNAAM", "geometry"]]
        RWZI_data.columns = ["RWZI_ID", "RWZI_identifier", "RWZI_name", "geometry"]
        
        # Pipe data
        pipe_data = gpd.read_file(path + "/" + "Leidingtrace.shp")
        pipe_data = pipe_data[["LDG_ID", "IDENTIFICA", "TRACE_NAAM", "STATUS", "geometry"]]
        pipe_data.columns = ["LDG_ID", "LD_identifier", "LD_name", "status", "geometry"]
        
        
        # Store data in class
        self.area_data = area_data
        self.RG_data = RG_data
        self.RWZI_regions = RWZI_regions
        self.RWZI_data = RWZI_data
        self.pipe_data = pipe_data


class get_file:   
    def save(self, obj, path):
        pickle.dump(obj, open(path + ".p", "wb" ))
    
    def load(self, path):
        return pickle.load(open(path + ".p", "rb" ))


class get_db:
    """
    This class can be used to automatically detect data in a given folder
    and transform it into a pickled file. This is done at initialization.
    
    ~~~~~ METHODS ~~~~~
    -- __init__()
    path           Location of the data.
    folder_tags    Names of folders that should be found from location.
    dump_path      Where to dump pickled files in.
    
    Returns None type object.
    
    -- load()
    path           Where to load pickled files from.
    tags           Names of pickled files.
    
    Returns tuple of data in order of tags.
    
    ~~~~~ EXAMPLE CALLS ~~~~~
    get_db(path=r"D:\DC3", folder_tags=["RG8180_L0", "data_pump_level"]) # Creating pickles
    data = get_db.load(path=r"D:\DC3\Full Project", tags=["RG8180_L0", "data_pump_level"])
    
    (!) IMPORTANT NOTE: CURRENTLY STORING THE PICKLE IS CORRUPTED THUS THE
    FUNCTION IS DEPRECATED.
    """
    def __init__(self, path: str=None, folder_tags: list=None, dump_path: str=None):
        
        # CREATE DIRECTORY IF IT DOESN'T EXIST
        if dump_path is None:
            dump_path=""
        else:
            if not os.path.exists(dump_path):
                os.makedirs(dump_path)
        
        # GET FOLDER LOCATIONS OF TAGS
        folder_locs = [utility.isolate_obj(utility.search_for(i, path, last_instance=True)) for i in folder_tags]
        
        # SAVE TAGGED DATA AS PICKLE
        for i, j in zip(folder_locs, folder_tags):
            if "knmi.harmonie_2018-01-01_2019-08-29" in j:
                data = get_rain(i, convert_time=True)
            elif "rain_grid_prediction" in j:
                data = get_rain_prediction(i, reduce_grid=True)
            else:
                data = load_all_pumps(i, convert_time=True)

            get_file().save(obj=data, path=dump_path + j)
    
    def load(path: str, tags: list):
        return tuple([get_file().load(path + "\\" + i) for i in tags])