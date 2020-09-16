import pandas as pd
import numpy as np
import utility
from scipy.signal import find_peaks
import datetime

rg_spots = \
{"Drunen":            (51.680344, 5.132245),
 "Haarsteeg":         (51.711222, 5.200918),
 "Elshout":           (51.700495, 5.142846),
 "WWTP":              (51.720422, 5.281040),
 "Bokhoven":          (51.736875, 5.233440),
 "Engelen":           (51.720494, 5.272021),
 "Maaspoort":         (51.719882, 5.290690),
 "Helftheuvelweg":    (51.700226, 5.269245),
 "Oude Engelenseweg": (51.697353, 5.290440),
 "Rompert":           (51.711202, 5.311724)}


def clean_mes_data(df, convert_timestamp=True, sort_timestamp=True, remove_duplicates=True, select_quality=True):
    '''
    This function convert the timestamp column to timestamp, sort on the timestamp column,
    removes duplicates and saves only the data with quality equal to 1
    '''
    if convert_timestamp:
        if df["TimeStamp"].dtype != "<M8[ns]":
            df["TimeStamp"] = pd.to_datetime(df["TimeStamp"])

    if sort_timestamp:
        df.sort_values("TimeStamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

    if remove_duplicates:
        df = df.loc[~df["TimeStamp"].duplicated()].reset_index(drop=True)

    if select_quality:
        df = df.loc[df["DataQuality"] == 1].reset_index(drop=True)

    return df


def merge_flow_level(flow_data, level_data):
    # INTERPOLATION OF MISSING MEASUREMENTS
    # Get all timestamps
    unique_timestamps = pd.concat([flow_data["TimeStamp"], level_data["TimeStamp"]]).unique()

    # Add timestamps to index
    flow_data = flow_data.set_index("TimeStamp").reindex(unique_timestamps)\
                         .reset_index(drop=False).sort_values("TimeStamp").reset_index(drop=True)
    level_data = level_data.set_index("TimeStamp").reindex(unique_timestamps)\
                           .reset_index(drop=False).sort_values("TimeStamp").reset_index(drop=True)

    return flow_data, level_data


def fill_flow(flow_data):
    '''
    Fill in missing flow data
    '''
    flow_data.loc[flow_data["Value"].isna() &\
                  (~flow_data["Value"].isna()).shift(1) &\
                  (~flow_data["Value"].isna()).shift(-1), "Value"] = flow_data["Value"].shift()\
                                                                            [flow_data["Value"].isna() &\
                                                                             (~flow_data["Value"].isna()).shift(1) &\
                                                                             (~flow_data["Value"].isna()).shift(-1)]
    flow_data["Value"] = flow_data["Value"].fillna(0)

    return flow_data


def fill_level(level_data):
    '''
    Fill missing level data
    '''
    na_indices = level_data.index[level_data["Value"].isna()]
    non_na_indices = level_data.index[~level_data["Value"].isna()]

    prior_indices = utility.search_prior_indices(na_indices, non_na_indices).reset_index(drop=True)
    posterior_indices = utility.search_posterior_indices(na_indices, non_na_indices).reset_index(drop=True)

    # TimeStamps of prior and posterior indices
    ts_prior = level_data.loc[prior_indices, "TimeStamp"]\
                         .apply(lambda i: (i - datetime.datetime(2017,1,1)).total_seconds())\
                         .reset_index(drop=True)
    ts_posterior = level_data.loc[posterior_indices, "TimeStamp"]\
                             .apply(lambda i: (i - datetime.datetime(2017,1,1)).total_seconds())\
                             .reset_index(drop=True)
    ts_actual = level_data.loc[level_data["Value"].isna(), "TimeStamp"]\
                          .apply(lambda i: (i - datetime.datetime(2017,1,1)).total_seconds())\
                          .reset_index(drop=True)

    # Levels of prior and posterior indices
    level_prior = level_data.loc[prior_indices, "Value"].reset_index(drop=True)
    level_posterior = level_data.loc[posterior_indices, "Value"].reset_index(drop=True)

    # Calculating weighted level values
    fill_values = (level_prior*(ts_posterior-ts_actual) + level_posterior*(ts_actual-ts_prior)) /\
                  (ts_posterior - ts_prior)
    fill_values.index = level_data.index[level_data["Value"].isna()]
    level_data["Value"] = level_data["Value"].fillna(fill_values)

    return level_data


def flow_group(flow):
    lst = flow.copy()

    not_equal_0 = lst != 0
    after_0 = lst.shift(1) == 0
    not_equal_and_after_0 = not_equal_0 & after_0
    not_equal_and_after_0 = not_equal_and_after_0.cumsum()
    lst.loc[lst != 0] = not_equal_and_after_0.loc[lst != 0]
    return lst.astype(int)


def level_group(lst):
    output = np.repeat(0, len(lst))

    maxima = find_peaks(lst, prominence=0.5)[0]
    minima = find_peaks(-lst, prominence=0.5)[0]

    min_indices = list(map(lambda i: np.where(minima > i)[0][0], maxima))

    for i, j, k in zip(maxima, minima[min_indices], range(len(maxima))):
        output[i:(j+1)] = k+1

    return pd.Series(output, index=lst.index)


def cell_index(x, y, reduced=False):
    x_space = np.linspace(start = -0.0185, stop = -0.0185 + 300*0.037, num = 301)
    y_space = np.linspace(start = 48.9885, stop = 48.9885 + 300*0.023, num = 301)

    x_out = np.where(x >= x_space)[0][-1]
    y_out = 300 - np.where(y <= y_space)[0][0]

    if reduced:
        x_out = x_out - 91
        y_out = y_out - 101

    return x_out, y_out

vec_cell_index = np.vectorize(cell_index)


def summarize_rain_data(rain_data, area_data=None, village_code=None, dry_threshold=0):
    """
    Function to reshape rain data to be fit for the DWAAS analysis.

    ~~~~~ INPUT  ~~~~~
    rain_data:     File as gathered by load_files.get_rain(...)
    area_data:     File as gathered by load_files.sdf(...).area_data
    village_code:  Identifier of the pump (e.g. 'DRU' for Drunen)
    dry_threshold: Minimum average rain per hour in the area that counts as wet
                   (2.5 recommended)

    ~~~~~ OUTPUT ~~~~~
    A data frame with the columns
    Date :     Date of measurement
    Total:     Average rainfall measurement in the area (unweighted by area size)
    DrySeries: Number of days since last rainfall.
    """

    # Convert to datetime if necessary
    if rain_data["Start"].dtype != "<M8[ns]":
            rain_data["Start"] = pd.to_datetime(rain_data["Start"])

    # Sort data by time because of it being possibly unordered
    rain_data.sort_values("Start", inplace=True)
    rain_data.reset_index(drop=True, inplace=True)

    # Selects only data from certain right village_code
    if village_code is not None:
        area_data["village_ID"] = area_data["sewer_system"].str.slice(4,7)
        area_data = area_data.loc[area_data["village_ID"] == village_code]
        areas = area_data["area_name"][area_data["area_name"].apply(lambda i: i in rain_data.columns)].to_list()

        rain_data = rain_data.loc[:, ["Start", "End"] + areas]

    # Create date column and sum up rain measurements over all area
    rain_data["Date"] = rain_data["Start"].apply(lambda i: i.date())
    rain_data["Total"] = rain_data.iloc[:, 2:].mean(axis=1)

    # Sum measurements by date and create dry-series column
    rain_data = rain_data.groupby("Date")["Total"].sum().reset_index(drop=False)
    rain_data["DrySeries"] = utility.reset_cumsum(rain_data["Total"], dry_threshold)

    return rain_data


def grid_area(rain_grid, rg: str, padding=1, reduced=False):
    
    x, y = cell_index(rg_spots[rg][1], rg_spots[rg][0], reduced=reduced)
    
    coords = (x - padding, x + padding, y - padding, y + padding)
    print(coords)
    
    return rain_grid[:, coords[0]:(coords[1]+1), coords[2]:(coords[3]+1)]


def flow_by_hour(df, impute_range=False):
    '''
    Calculates the total amount of flow per hour
    '''
    flow_data = df.copy()

    flow_data["TimeSpan"] = flow_data["TimeStamp"].diff(1).apply(lambda i: i.seconds).fillna(5)
    flow_data["TimeHour"] = flow_data["TimeStamp"].apply(lambda i: i.replace(minute=0, second=0))
    flow_data["Flow"] = flow_data["Value"] / 3600 * flow_data["TimeSpan"]

    flow_data = flow_data.groupby("TimeHour").aggregate({"Flow": np.sum, "DataQuality": np.mean, "TimeSpan": np.sum})
    
    if impute_range:
        dt_range = pd.date_range(flow_data.index[0].floor('h'), flow_data.index[-1].floor('h'), freq='h')
        flow_data = flow_data.reindex(dt_range)
        flow_data["Flow"] = flow_data["Flow"].fillna(0)
        flow_data["DataQuality"] = flow_data["DataQuality"].fillna(0)
    
    return flow_data.reset_index(drop=False).rename(columns={"index": "TimeHour"})


def match_by_timestamp(rain_prediction, hourly_flow, multiple=False, steps=3):
    """
    match the rain_prediction with the hourly_flow, by timestamp
    """
    # Omitting all negative rain predictions
    rain_prediction[1][rain_prediction[1] < 0] = 0
    
    if multiple:
        bool_1 = np.sum([(hourly_flow["TimeHour"]-pd.Timedelta(hours=i)).isin(rain_prediction[0]["start"].shift(i))
                         for i in range(steps)]
                        , axis=0) > 1
        hourly_flow_indices = hourly_flow["TimeHour"][bool_1]
        
    else:
        bool_1 = hourly_flow["TimeHour"].isin(rain_prediction[0]["start"])
        hourly_flow_indices = hourly_flow["TimeHour"][bool_1]
    
    # Select common TimeStamps
    shared_indices = np.intersect1d(rain_prediction[0]["start"].values, hourly_flow_indices.values)
    
    # Get integer indices to use for rain_prediction
    rp_indices = rain_prediction[0]["start"].reset_index(drop=False)\
                                            .set_index("start")\
                                            .reindex(shared_indices).values.flatten()
    
    if multiple:
        grid = np.stack([rain_prediction[1][rp_indices-i] for i in range(steps)])
    
        grid = grid.reshape(grid.shape[0], grid.shape[1], grid.shape[2] * grid.shape[3])
        grid = np.concatenate(grid, axis=1)
    else:
        grid = rain_prediction[1][rp_indices]
        grid = grid.reshape(grid.shape[0], grid.shape[1] * grid.shape[2])
    
    return grid, hourly_flow.set_index("TimeHour").reindex(shared_indices).reset_index(drop=False)
