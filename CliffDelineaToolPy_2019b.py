"""
# CliffDelineaToolPy v1.2.0
# An algorithm to map coastal cliff base and top from topography
# https://github.com/zswirad/CliffDelineaTool
# Zuzanna M Swirad (zswirad@ucsd.edu), Scripps Institution of Oceanography, \
# UC San Diego
# Help in debugging: George Thomas
# Originally coded in MATLAB 2019a (v1.2.0; 2021-11-24)
# Last updated on 2022-1-24 (Python 3.8)
# Modified by Collin Roland 2024-02-03
# To do: Improve functionalization and parallelize
"""

# %% Import packages
import geopandas as gpd
import glob
import math
import numpy as np
import os
from pathlib import Path
import pandas as pd
import statsmodels.api as sm
import time
from joblib import Parallel, delayed

np.seterr(divide='ignore', invalid='ignore')
# %% Set variables

# Indicate input data parameters:
data_path = r"D:\CJR\LakeSuperior_BluffDelin\2019\delineation_points_text"
bluff_top_dir = os.path.abspath(os.path.join(data_path, r'..', 'delineation_top_points'))
bluff_base_dir = os.path.abspath(os.path.join(data_path, r'..', 'delineation_base_points'))

# Set the calibrated input variables:
local_scale = 5  # How many adjacent points to consider as a local scale?
base_max_elev = 184.5 # What is the top limit for cliff base elevation (m)?
base_sea_slope = 3  # What is the max seaward slope for cliff base (deg)?
base_land_slope = 5  # What is the min landward slope for cliff base (deg)?
top_sea_slope = 10  # What is the min seaward slope for cliff top (deg)?
top_land_slope = 5  # What is the max landward slope for cliff top (deg)?
prop_convex = 0.15  # What is the minimal proportion of the distance from trendline #2 to replace modelled cliff top location?
smooth_window = 3  # What is the alongshore moving window (number of transects) for cross-shore smoothing (points)? INTEGER
proj_crs = 'EPSG:32615'
params_strict = {"local_scale": local_scale, 
          "base_max_elev": base_max_elev,
          "base_sea_slope": base_sea_slope, 
          "base_land_slope": base_land_slope,
          "top_sea_slope": top_sea_slope, 
          "top_land_slope": top_land_slope,
          "prop_convex": prop_convex, 
          "smooth_window": smooth_window, 
          "proj_crs": proj_crs}

# Relaxed parameter set for failed top ID
params_lax = {"local_scale": local_scale, 
          "base_max_elev": base_max_elev,
          "base_sea_slope": base_sea_slope, 
          "base_land_slope": base_land_slope-2,
          "top_sea_slope": top_sea_slope-10, 
          "top_land_slope": top_land_slope+10,
          "prop_convex": prop_convex, 
          "smooth_window": smooth_window, 
          "proj_crs": proj_crs}

# %% Self-defined functions


def process_transect(data, num_transect, transects_mod, params):
    # Debugging
    # n = 0
    start_transect = time.time()
    print('Processing ', num_transect, ' out of ', max(data['TransectID']), ' transects.')
    sub = data[data['TransectID'] == num_transect]  # subset points from single transect
    row_count = sub.shape[0]  # count of points in transect
    if row_count > 0:  # process transect only if there are data
        sub = sub.sort_values(['Distance'])  # sort values by distance from seaward end of transect
        sub = sub.reset_index(drop=True)  # reindex
        # Fill data gaps:
        sub.loc[sub['Elevation'] < -50, ['Elevation']] = np.nan
        sub = sub.interpolate()
        sub = sub.ffill()
        sub = sub.bfill()
        # Calculate local slopes:
        zeros = np.zeros(row_count + 1)
        sub['SeaSlope'] = pd.Series(zeros)  # seaward slope (average slope between the point and nVert consecutive seaward points)
        sub['LandSlope'] = pd.Series(zeros)  # landward slope (average slope between the point and nVert consecutive landward points)
        sub['Trendline1'] = pd.Series(zeros)  # trendline #1
        sub['Difference1'] = pd.Series(zeros)  # elevations - trendline #1
        sub = sub.fillna(0)

        # Calculate slopes iterating across local scales window
        start_local = time.time()  # set start time for local scale processing
        for local_point in range(local_scale, row_count - params["local_scale"]):  # Array from local_scale (10) to transect_length (90) minus local scale

            # seaward slopes
            count = 0
            for local_scale_window_point in range(1, params["local_scale"] + 1):
                loc_pt_2 = local_point - local_scale_window_point
                if sub.Elevation[local_point] != sub.Elevation[loc_pt_2]:
                    angle = math.degrees(math.atan((sub.Elevation[local_point]
                                                    - sub.Elevation[loc_pt_2])
                                                   / (sub.Distance[local_point]
                                                      - sub.Distance[loc_pt_2])))
                    if angle < 0:
                        angle = 0
                    sub.loc[local_point, 'SeaSlope'] = sub.loc[local_point, 'SeaSlope'] + angle  # Cumulative sum of sea slope angles across local scale window
                    count += 1

            sub.loc[local_point, 'SeaSlope'] = sub.loc[local_point, 'SeaSlope'] / count  # average across viable sea slope angles

            # landward slopes
            count = 0
            for local_scale_window_point in range(1, params["local_scale"] + 1):
                loc_pt_2 = local_point - local_scale_window_point
                loc_pt_3 = local_point + local_scale_window_point
                if sub.Elevation[local_point] != sub.Elevation[loc_pt_2]:
                    angle = math.degrees(math.atan((sub.Elevation[loc_pt_3]
                                                    - sub.Elevation[local_point]) /
                                                   (sub.Distance[loc_pt_3]
                                                    - sub.Distance[local_point])))
                    if angle < 0:
                        angle = 0
                    sub.loc[local_point, 'LandSlope'] = sub.loc[local_point, 'LandSlope'] + angle
                    count += 1

            sub.loc[local_point, 'LandSlope'] = sub.loc[local_point, 'LandSlope'] / count

        print("Local scale processing took", time.time() - start_local, ' seconds.')

        # Limit the transect landwards to the highest point + local_scale:
        ind_max = np.argmax(sub['Elevation'], axis=0)
        if row_count > ind_max + local_scale:
            all_drop = row_count - (ind_max + local_scale + 1)
            sub.drop(sub.tail(all_drop).index, inplace=True)
        row_count = sub.shape[0]  # calculated updated transect length
        sub = sub.reset_index(drop=True)

        # Draw trendline #1 (straight line between the seaward and landward transect ends):
        sub.loc[0, 'Trendline1'] = sub.loc[0, 'Elevation']  # set seaward trendline elevation
        sub.iloc[-1, sub.columns.get_loc('Trendline1')] = sub.iloc[-1, sub.columns.get_loc('Elevation')]  # set landward trendline elevation
        for local_point in range(1, row_count):  # Loop over length of transect calculating linear trendline
            sub.loc[local_point, 'Trendline1'] = ((sub.loc[local_point, 'Distance'] - sub.loc[0, 'Distance'])
                                                  * (sub.iloc[-1, sub.columns.get_loc('Elevation')] - sub.loc[0, 'Elevation'])
                                                  / (sub.iloc[-1, sub.columns.get_loc('Distance')] - sub.loc[0, 'Distance'])
                                                  + sub.loc[0, 'Elevation'])

        # Calculate vertical distance between actual elevations and trendline #1:
        sub['Difference1'] = sub['Elevation'] - sub['Trendline1']

        transects_mod = pd.concat([transects_mod, sub])  # Append modified transect to table of transects
        print("Transect ", num_transect, ' processed in ', time.time() - start_transect, ' seconds.')
        return transects_mod


def identify_potential_base_points(potential_base, modelled_base):
    if potential_base.shape[0] > 0:  # Only identify base points if potential points exist
        cliffed_profiles = potential_base['TransectID'].unique()  # Identify transect ID's that contain potential base points
        for n in range(potential_base['TransectID'].min(), potential_base['TransectID'].max() + 1):
            for m in range(cliffed_profiles.shape[0]):
                if n == cliffed_profiles[m]:
                    # Debugging
                    # n = 10
                    sub = potential_base[potential_base['TransectID'] == n]
                    sub = sub.sort_values(by=['Difference1'])
                    modelled_base = pd.concat([modelled_base, sub.iloc[[0]]])
    return modelled_base, cliffed_profiles


def identify_potential_top_points(transects_mod, transect_id, modelled_base, modelled_top, cliffed_profiles, params):
    for m in range(cliffed_profiles.shape[0]):
        if transect_id == cliffed_profiles[m]:
            # Debugging
            # n=10
            sub = transects_mod[transects_mod['TransectID'] == transect_id]
            sub = sub.reset_index(drop=True)

            # Remove points seawards from the cliff base:
            sub_base = modelled_base[modelled_base['TransectID'] == transect_id]
            sub_base = sub_base.reset_index(drop=True)
            sub_base_dist = sub_base.Distance[0]
            sub.drop(sub[sub['Distance'] < sub_base_dist].index, inplace=True)
            sub = sub.reset_index(drop=True)

            # Draw trendline #2 between cliff base and landward transect end:
            row_count = sub.shape[0]
            zeros = np.zeros(row_count + 1)
            sub['Trendline2'] = pd.Series(zeros)  # trendline #2
            sub['Difference2'] = pd.Series(zeros)  # elevation - trendline #2
            sub = sub.fillna(0)

            sub.loc[0, 'Trendline2'] = sub.loc[0, 'Elevation']
            sub.iloc[-1, sub.columns.get_loc('Trendline2')] = sub.iloc[-1, sub.columns.get_loc('Elevation')]
            for local_point in range(1, row_count):
                sub.loc[local_point, 'Trendline2'] = ((sub.Distance[local_point] - sub.Distance[0])
                                                      * (sub.iloc[-1, sub.columns.get_loc('Elevation')] - sub.Elevation[0])
                                                      / (sub.iloc[-1, sub.columns.get_loc('Distance')] - sub.Distance[0])
                                                      + sub.Elevation[0])
            sub['Difference2'] = sub['Elevation'] - sub['Trendline2']

            # Find potential cliff top locations:
            potential_top = sub[(sub['SeaSlope'] > params["top_sea_slope"]) & (sub['LandSlope'] < params["top_land_slope"]) & (sub['Difference2'] > 0)]
            
            if potential_top.shape[0] > 0:
                potential_top = potential_top.sort_values(by=['Difference2'])
                
                # From the points that satisfy the criteria, for each transect select one with the largest vertical difference between the elevation and trendline #2:
                modelled_top0 = potential_top.iloc[[-1]]   
                
                # Check whether the selected point is part of within-cliff flattening:
                if (potential_top['Distance'].max() > (modelled_top0.Distance.values[0] + local_scale)):
                    sub_new = sub.copy()
                    sub_top_dist = potential_top.iloc[-1, sub.columns.get_loc('Distance')]
                    sub_new.drop(sub_new[sub_new['Distance'] < sub_top_dist].index, inplace=True)  # remove points seawards from the modelled cliff top
                    row_count_new = sub_new.shape[0]
                    zeros_new = np.zeros(row_count_new + 1)
                    sub_new['Trendline3'] = pd.Series(zeros_new)
                    sub_new['Difference3'] = pd.Series(zeros_new)
                    sub_new = sub_new.fillna(0)
                    sub_new = sub_new.reset_index(drop=True)

                    sub_new.loc[0, 'Trendline3'] = sub_new.loc[0, 'Elevation']
                    sub_new.iloc[-1, sub_new.columns.get_loc('Trendline3')] = sub_new.iloc[-1, sub_new.columns.get_loc('Elevation')]
                    for local_point in range(1, row_count_new):
                        sub_new.loc[local_point, 'Trendline3'] = ((sub_new.Distance[local_point] - sub_new.Distance[0])
                                                                  * (sub_new.iloc[-1, sub_new.columns.get_loc('Elevation')] - sub_new.Elevation[0])
                                                                  / (sub_new.iloc[-1, sub_new.columns.get_loc('Distance')] - sub_new.Distance[0])
                                                                  + sub_new.Elevation[0])
                    sub_new['Difference3'] = sub_new['Elevation'] - sub_new['Trendline3']

                    potential_top2 = potential_top.copy()
                    potential_top2.drop(potential_top2[potential_top2['Distance'].values < modelled_top0.Distance.values].index, inplace=True)
                    row_count_new = potential_top2.shape[0]
                    zeros_new = np.zeros(row_count_new + 1)
                    potential_top2['Difference3'] = pd.Series(zeros_new)
                    potential_top2 = potential_top2.fillna(0)
                    potential_top2 = potential_top2.reset_index(drop=True)

                    for p in range(potential_top2.shape[0]):
                        sub_new_temp = sub_new[sub_new['Distance'] == potential_top2.Distance[p]]
                        potential_top2.iloc[p, potential_top2.columns.get_loc('Difference3')] = sub_new_temp.Difference3

                    potential_top2 = potential_top2[((potential_top2['Difference3'].values > 0)
                                                     & (potential_top2['Difference2'].values >=
                                                        modelled_top0.Difference2.values * prop_convex)
                                                     & (potential_top2['Distance'].values >= modelled_top0.Distance.values + local_scale))]
                    if potential_top2.shape[0] > 0:
                        potential_top2 = potential_top2.sort_values(by=['Difference2'])
                        potential_top2.drop(['Difference3'], axis=1)
                        modelled_top0 = potential_top2.iloc[[-1]]
                    
                modelled_top = pd.concat([modelled_top, modelled_top0])  
    return modelled_top


def fix_outlier_top_points(transects_mod, modelled_base, modelled_top, row_count, params):
    if (row_count >= params["smooth_window"]):
        model = sm.OLS(modelled_top['Distance'], modelled_top['SmoothedDistance'])
        results = model.fit()
        influence = results.get_influence()
        modelled_top['StandResidual'] = influence.resid_studentized_internal
        modelled_top.loc[abs(modelled_top['StandResidual']) > 2, ['Outlier']] = 1
        fix = modelled_top[modelled_top['Outlier'] == 1]
        # 2. Delete or replace outliers with more suitable potential cliff tops:
        # (Repeat cliff top detection for the transects with outliers.)
        if fix.shape[0] > 0:
            fix = fix.reset_index(drop=True)
            modelled_top.drop(['StandResidual', 'Outlier'], axis=1)
            for c in range(fix.shape[0]):
                sub = transects_mod[transects_mod['TransectID'] == fix.TransectID[c]]
                sub = sub.reset_index(drop=True)
                outlier = modelled_top[modelled_top['TransectID'] == fix.TransectID[c]]
                # Remove points seawards from the cliff base:
                sub_base = modelled_base[modelled_base['TransectID'] == fix.TransectID[c]]
                sub_base = sub_base.reset_index(drop=True)
                sub_base_dist = sub_base.Distance[0]
                sub.drop(sub[sub['Distance'] < sub_base_dist].index, inplace=True)
                sub = sub.reset_index(drop=True)

                # Draw trendline #2 between cliff base and landward transect end:
                rowCount = sub.shape[0]
                zeros = np.zeros(rowCount + 1)
                sub['Trendline2'] = pd.Series(zeros)  # trendline #2
                sub['Difference2'] = pd.Series(zeros)  # elevation - trendline #2
                sub = sub.fillna(0)

                sub.loc[0, 'Trendline2'] = sub.loc[0, 'Elevation']
                sub.iloc[-1, sub.columns.get_loc('Trendline2')] = sub.iloc[-1, sub.columns.get_loc('Elevation')]
                for z in range(1, rowCount):
                    sub.loc[z, 'Trendline2'] = ((sub.Distance[z] - sub.Distance[0])
                                                * (sub.iloc[-1, sub.columns.get_loc('Elevation')] - sub.Elevation[0])
                                                / (sub.iloc[-1, sub.columns.get_loc('Distance')] - sub.Distance[0])
                                                + sub.Elevation[0])
                sub['Difference2'] = sub['Elevation'] - sub['Trendline2']

                # Find potential cliff top locations:
                potential_top = sub[(sub['SeaSlope'] > params["top_sea_slope"]) & (sub['LandSlope'] < params["top_land_slope"]) & (sub['Difference2'] > 0)]
                row_count = potential_top.shape[0]     
                zeros = np.zeros(row_count + 1)
                potential_top['SmoothedDistance'] = pd.Series(zeros)  # smoothed distance
                potential_top['DistanceFromSmoothed'] = pd.Series(zeros)  # distance from smoothed distance

                potential_top = potential_top.fillna(0)
                potential_top['SmoothedDistance'] = fix.SmoothedDistance[c]
                potential_top['DistanceFromSmoothed'] = abs(potential_top['Distance'] - potential_top['SmoothedDistance'])
                potential_top = potential_top.sort_values(by=['DistanceFromSmoothed'])
                potential_top = potential_top.iloc[0]

                modelled_top.loc[(modelled_top['TransectID'] == potential_top['TransectID']), 'PointID'] = potential_top['PointID']
                modelled_top.loc[(modelled_top['TransectID'] == potential_top['TransectID']), 'Elevation'] = potential_top['Elevation']
                modelled_top.loc[(modelled_top['TransectID'] == potential_top['TransectID']), 'Distance'] = potential_top['Distance']
                modelled_top.loc[(modelled_top['TransectID'] == potential_top['TransectID']), 'SeaSlope'] = potential_top['SeaSlope']
                modelled_top.loc[(modelled_top['TransectID'] == potential_top['TransectID']), 'LandSlope'] = potential_top['LandSlope']
                modelled_top.loc[(modelled_top['TransectID'] == potential_top['TransectID']), 'Trendline1'] = potential_top['Trendline1']
                modelled_top.loc[(modelled_top['TransectID'] == potential_top['TransectID']), 'Difference1'] = potential_top['Difference1']
                modelled_top.loc[(modelled_top['TransectID'] == potential_top['TransectID']), 'Trendline2'] = potential_top['Trendline2']
                modelled_top.loc[(modelled_top['TransectID'] == potential_top['TransectID']), 'Difference2'] = potential_top['Difference2']

            row_count = modelled_top.shape[0]
            zeros = np.zeros(row_count + 1)
            modelled_top['StandResidual'] = pd.Series(zeros)  # standardized residuals
            modelled_top['Outlier'] = pd.Series(zeros)  # outliers
            modelled_top = modelled_top.fillna(0)

            model = sm.OLS(modelled_top['Distance'], modelled_top['SmoothedDistance'])
            results = model.fit()
            influence = results.get_influence()
            modelled_top['StandResidual'] = influence.resid_studentized_internal
            modelled_top.loc[abs(modelled_top['StandResidual']) > 2, ['Outlier']] = 1

            modelled_top.drop(modelled_top[modelled_top['Outlier'] == 1].index, inplace=True)  # ignore new cliff top positions if standardized residuals did not improve
    return modelled_top
# %% Big function

def delineate_cliff_features(data_path, number, fileName, out_bluff_top_dir,
                             out_bluff_base_dir, params_strict, params_lax):
    # Debugging
    # fileName = glob.glob('*.txt')[35]
    # with open(fileName, 'r') as fin:
    #     data = fin.read().splitlines(True)
    # with open(fileName, 'w') as fout:
    #     fout.writelines(data[1:])
    os.chdir(data_path)  # Change to directory of point *.txt files
    data = pd.read_csv(fileName, header=0)  # Read point file for a single tile
    data.columns = ['PointID', 'TransectID', 'Elevation', 'Distance', 'Easting', 'Northing']  # Define column names
    transects_mod = pd.DataFrame()  # Create empty dataframe for modified transect data

    # Pre-processing all transects in a single tile (calculate slopes and trendlines)
    for num_transect in range(min(data['TransectID']), max(data['TransectID']) + 1):
        transects_mod = process_transect(data, num_transect, transects_mod, params_strict)
    transects_mod = transects_mod.reset_index(drop=True)

    # Find potential cliff base locations:
    potential_base = transects_mod[(transects_mod['Elevation'] < params_strict["base_max_elev"])
                                   & (transects_mod['SeaSlope'] < params_strict["base_sea_slope"])
                                   & (transects_mod['LandSlope'] > params_strict["base_land_slope"])
                                   & (transects_mod['Difference1'] < 0)]

    # From the points that satisfy the criteria, for each transect select one with the largest vertical difference between the elevation and trendline #1:
    modelled_base = pd.DataFrame(columns=potential_base.columns)
    modelled_base, cliffed_profiles = identify_potential_base_points(potential_base, modelled_base)

    # Find cliff top locations for transects with a cliff base point:
    if modelled_base.shape[0] > 0:
        modelled_top = pd.DataFrame()

        for transect_id in range(int(modelled_base['TransectID'].min()), int(modelled_base['TransectID'].max() + 1)):
            modelled_top = identify_potential_top_points(transects_mod, transect_id,
                                                         modelled_base, modelled_top, cliffed_profiles, params_strict)

        if modelled_top.shape[0] < 0:
            for transect_id in range(int(modelled_base['TransectID'].min()), int(modelled_base['TransectID'].max() + 1)):
                modelled_top = identify_potential_top_points(transects_mod, transect_id,
                                                             modelled_base, modelled_top, cliffed_profiles, params_lax)
        # Save the base data
        os.chdir(out_bluff_base_dir)
        modelled_base = modelled_base.sort_values(by=['TransectID'])
        save_name_base = fileName[:-4] + '_base.shp'
        modelled_base_save = modelled_base[['PointID', 'TransectID', 'Easting', 'Northing']]  # Select which columns to save; you may want to add XY coordinates if they were present
        modelled_base_save = gpd.GeoDataFrame(modelled_base_save, geometry=gpd.points_from_xy(modelled_base_save.Easting, modelled_base_save.Northing), crs=params_strict["proj_crs"])
        modelled_base_save.to_file(save_name_base)  # change to header=True if exporting with header

        # Remove alongshore outliers:
        # 1. Find outliers:
        try:
            modelled_top = modelled_top.sort_values(by=['TransectID'])
            row_count = modelled_top.shape[0]
            zeros = np.zeros(row_count + 1)
            modelled_top['SmoothedDistance'] = pd.Series(zeros)  # smoothed distance
            modelled_top['StandResidual'] = pd.Series(zeros)  # standardized residuals
            modelled_top['Outlier'] = pd.Series(zeros)  # outliers (https://urldefense.proofpoint.com/v2/url?u=https-3A__online.stat.psu.edu_stat462_node_172_&d=DwIGAg&c=-35OiAkTchMrZOngvJPOeA&r=8yfrSqW1K1RIJJQehgvwMvTlPVMycUwQP0bc0m2ZrpA&m=FzFRg9yDDPqUWGYFkZANybMJTnJ5ceO8bU_NZFIOtTnG0rReObfDNmi7RpBlydEv&s=7mLUlYzS0mWlfKq6l4iPPJU4nGwCicT73n_ue03hzaA&e= ; accessed on 2021/06/04)
            modelled_top = modelled_top.fillna(0)
            modelled_top['SmoothedDistance'] = modelled_top['Distance'].rolling(window=params_strict["smooth_window"]).median()
            modelled_top['SmoothedDistance'] = modelled_top['SmoothedDistance'].fillna(method='ffill')
            modelled_top['SmoothedDistance'] = modelled_top['SmoothedDistance'].fillna(method='bfill')
    
            modelled_top = fix_outlier_top_points(transects_mod, modelled_base, modelled_top, row_count, params_strict)
    
            # Save the top data:
            if modelled_top.shape[0] > 0:
                os.chdir(out_bluff_top_dir)
                save_name_top = fileName[:-4] + '_top.shp'
                modelled_top_save = modelled_top[['PointID', 'TransectID', 'Easting', 'Northing']]  # Select which columns to save; you may want to add XY coordinates if they were present
                modelled_top_save = gpd.GeoDataFrame(modelled_top_save, geometry=gpd.points_from_xy(modelled_top_save.Easting, modelled_top_save.Northing), crs=params_strict["proj_crs"])
                modelled_top_save.to_file(save_name_top)  # change to header=True if exporting with header
    
            print("File ", fileName, ' processed in ', time.time() - start_file, ' seconds.')
        except KeyError:
            print(r'Key Error - suspect no top points delineated')

# %% Loop over transect data files

# cache_dir = Path(r'D:\CJR\LakeSuperior_BluffDelin\2009\mem_cache')
# from joblib import Memory
# memory = Memory(cache_dir, verbose=0)
out_bluff_top_dir = Path(r'D:\CJR\LakeSuperior_BluffDelin\2019\delineation_top_points')
out_bluff_base_dir = Path(r'D:\CJR\LakeSuperior_BluffDelin\2019\delineation_base_points')
os.chdir(r"D:\CJR\LakeSuperior_BluffDelin\2019\delineation_points_text")
for number, fileName in enumerate(glob.glob('*.txt')):
    if number >= 0:
        start_file = time.time()
        delineate_cliff_features(data_path, number, fileName, out_bluff_top_dir, out_bluff_base_dir, params_strict, params_lax)
# %%

top_points = pd.read_csv(saveName2,header=None)
top_points.columns =['PointID', 'TransectID','Easting','Northing'] 
base_points = pd.read_csv(saveName1, header=None)
base_points.columns =['PointID', 'TransectID','Easting','Northing']
top_points['geometry']=[Point(i[1][2],i[1][3]) for i in top_points.iterrows()]
base_points['geometry']=[Point(i[1][2],i[1][3]) for i in base_points.iterrows()]
top_points = gpd.GeoDataFrame(data=top_points,crs='EPSG:6344')
base_points = gpd.GeoDataFrame(data=base_points,crs='EPSG:6344')
base_points.to_file('base_points.shp')
top_points.to_file('top_points.shp')
