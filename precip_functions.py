#=================================
# This Python defines different functions for analyzing the precipitation data
# Edited by Yi Hong, Oct. - Nov. 2019
#=================================
#%% import packages
import numpy as np
import pandas as pd
#import xarray as xr
import datetime
import os
import re
from math import cos, asin, sqrt

#%% =============================================================================
# Define a function to read the rain gauge data from generated files
# Return a list with the required data
# =============================================================================
def Read_rain_file(record_dic, gauge_file, begin_date, end_date, date_format):
#    dir_ReCON=record_dic+'ReCON_stations\\'  
    begin_time=datetime.datetime.strptime(begin_date, date_format)
    end_time=datetime.datetime.strptime(end_date,date_format)
    
    f=open(os.path.join(record_dic, gauge_file), 'r', errors=None)
    record_file=f.readlines()
    f.close()
    
    begin_line=record_file[0].split(';')           
    record_begin=datetime.datetime.strptime(str(begin_line[0]),date_format)
    end_line=record_file[-1].split(';')           
    record_end=datetime.datetime.strptime(str(end_line[0]), date_format)

    if begin_time<(record_begin-datetime.timedelta(days=366)):         # have to consider if record begins in the middle of a year
        print('Please reset a begin date')
        return []
    elif end_time>(record_end+datetime.timedelta(days=366)):
        print('Please reset an end date')   # have to consider if record ends in the middle of a year
        return []
    else:       
        i=0
        j=len(record_file)-1
        while datetime.datetime.strptime(str(record_file[i].split(';')[0]),date_format)<begin_time:
            i+=1
        while datetime.datetime.strptime(str(record_file[j].split(';')[0]),date_format)>end_time+ datetime.timedelta(days=1):
            j-=1
        return record_file[i:j]
        
        # test=[s for s in record_file if '2015/12/31' in s] ## cannot use this, cause the data is not complete
#%%
# =============================================================================
# Resample the precip record with the specified frequency of input and output data
# Set the limit for nan data        
# =============================================================================
def resamp_precip(freq_in, freq_out, resample_file, min_count, begin_date, end_date, date_format):
    #% it is better to use the pandas dataframe for the calculations
    df = pd.read_csv(resample_file,delimiter=';',names=['Time','Precip'])
    df['DateTime'] = pd.to_datetime(df['Time'],format=date_format)
    df=df.drop(['Time'],axis=1)
    #% Use a date time index from the beginning to the end at the specified time step       
#    record_begin=df['DateTime'].iloc[0]  
#    record_end=df['DateTime'].iloc[-1] 
    begin_time=datetime.datetime.strptime(begin_date,date_format)
    end_time=datetime.datetime.strptime(end_date,date_format)
    #(end_time-begin_time)/datetime.timedelta(minutes=30)
    #% Not necessary to test the begin time
#    if begin_time<record_begin or end_time>record_end:         # have to consider if record begins in the middle of a year
#        print('Please reset the dates')
#        return []
#    else:       
    #% Slice the data and set datetime index
    df_query=df.loc[(df['DateTime']>=begin_time) & (df['DateTime']<=end_time)].drop_duplicates()     #df.between_time
#    df_query[df_query.index.duplicated()]    
    df_query=df_query.set_index('DateTime')
    df_query=df_query.sort_index().to_period(freq_in)
    #print('shape of the analyzed dataframe: ',df_query.shape)
    #df_query.plot(y='Precip')
    #% Fill in with the full 30 min data
    rng_in = pd.period_range(begin_time,end_time,freq=freq_in)
    df_query=df_query.reindex(rng_in)
    na_in=df_query['Precip'].isna().sum()
    #df_query.asfreq(freq=freq_in, fill_value=-999)
    #print('Test of the precip sum: ',df_query['Precip'].sum())
    #print('Nans in the new precip serie: ',na_in)
    #%
    precip_out=df_query.resample(freq_out).sum(min_count=min_count)  # sum(skipna=true)
    na_out=precip_out['Precip'].isna().sum()
    #%
    return precip_out, na_in, na_out
#%% Resample a dataframe with specified frequency of input and output data
    ### the df is already pre-treated with a datetime index and sort
 
def resamp_df(df, freq_in, freq_out, min_count, begin_date, end_date, date_format):
    begin_time=datetime.datetime.strptime(begin_date,date_format)
    end_time=datetime.datetime.strptime(end_date,date_format)
    df_query=df.loc[(df.index>=begin_time) & (df.index<=end_time)]   #df.between_time   
    
    df_query=df_query.to_period(freq_in)
    rng_in = pd.period_range(begin_time,end_time,freq=freq_in)
    df_query=df_query.reindex(rng_in)
    na_in=df_query.iloc[:,0].isna().sum()
    precip_out=df_query.resample(freq_out).sum(min_count=min_count)  # sum(skipna=true)
    na_out=precip_out.iloc[:,0].isna().sum()
    return precip_out, na_in, na_out

#%%
# =============================================================================
# # fonction to select AORC data correspond to rain gauge stations
# =============================================================================
def AORC_raingauge(ds,index):
    if ds['RAINRATE'].size>0 and ds['valid_time'].size>0:
        rain_array=ds['RAINRATE']
        rain_intense=rain_array[dict(Time=0, south_north=index[0], west_east=index[1])].data # mm s^-1
        rain_time=ds['valid_time'].data
    return rain_time,rain_intense

#%%  
# =============================================================================
# Function to extract the date from the filename by using the regular expression to find        
# =============================================================================
def date_filename(filename):        
    date_format='20[01]\d[01]\d[0123]\d.nc'                                        # date format
#    date_format=set_year(year)
    date_nc=re.findall(date_format,filename)
    if len(date_nc)>0: 
        str_date=date_nc[0][0:len(date_nc[0])-3]        
        return str_date
    else:
        return []
# =============================================================================
# Functions to calculate the RMSE and MAE, and (prod-obs)/obs
# =============================================================================
def MAE_RMSE_Diff(serie1, serie2):
    if len(serie1)==len(serie2):
        serie1=serie1.reset_index(drop=True)
        serie2=serie2.reset_index(drop=True)
        
        MAE=(serie1-serie2).abs().mean()
        RMSE=np.sqrt(((serie1-serie2)**2).mean())
        Diff=(serie1-serie2).sum(min_count=1)
        return MAE, RMSE, Diff
    else:
        raise Exception('Length of the two series are not equal')
        return []


def PBIAS(serie1, serie2):  # this function is to calculate the Diff/Obs, the problem is when obs is low, this value may over estimate
    if len(serie1)==len(serie2):
        if serie1.isnull().all() or serie2.isnull().all():
            return np.nan
        else:
            Total_dif=0
            Total_obs=0
            for i in range(len(serie1)):
                if (not np.isnan(serie1.iloc[i]) and not np.isnan(serie2.iloc[i])):
                    Total_dif += serie1.iloc[i]-serie2.iloc[i]
                    Total_obs += serie2.iloc[i]        
            if Total_obs>0:
                PBIAS= 100 * Total_dif/Total_obs
                return PBIAS                
            else: # if Total_obs ==0, how to set the diff ?
#                return serie1.sum()
                return np.nan
    else:
        raise Exception('Length of the two series are not equal')
        return np.nan    

def R2(serie1, serie2):  # this function is to calculate the determinant R2
    if len(serie1)==len(serie2):
        if serie1.isnull().all() or serie2.isnull().all():
            return np.nan
        else:            # Compute only when both S1 and S2 is true
            S1=[]
            S2=[]            
            for i in range(len(serie1)):
                if (not np.isnan(serie1.iloc[i]) and not np.isnan(serie2.iloc[i])):
                    S1.append(serie1.iloc[i])
                    S2.append(serie2.iloc[i])       
            if len(S1)==0:
                return np.nan
            else:
                mean_S1=sum(S1)/len(S1)
                mean_S2=sum(S2)/len(S2)
                diff_S12=0
                diff2_S1=0
                diff2_S2=0  
                for j in range(len(S1)):
                    diff_S12 += (S1[j]-mean_S1)*(S2[j]-mean_S2)
                    diff2_S1 += (S1[j]-mean_S1)**2
                    diff2_S2 += (S2[j]-mean_S2)**2

                if diff2_S1 != 0 and diff2_S2 != 0:
                    R2 = diff_S12/(np.sqrt(diff2_S1)*np.sqrt(diff2_S2))
                    return R2
                else:
                    return np.nan
    else:
        raise Exception('Length of the two series are not equal')
        return np.nan  


    
# =============================================================================
# Function to plot the base map
# =============================================================================
def map_area(shorline_file):
    sl_coor= pd.read_csv(shorline_file, sep=r'\s+', skiprows=2, header=None, index_col=False, skipinitialspace=True,engine='python', names=['Lon','Lat'])
    min_Lon=min(sl_coor['Lon']) 
    max_Lon=max(sl_coor['Lon'].loc[sl_coor['Lon']<0])
    min_Lat=min(sl_coor['Lat'].loc[sl_coor['Lat']>40])
    max_Lat=max(sl_coor['Lat'].loc[sl_coor['Lat']<50])    
    return min_Lon,max_Lon,min_Lat,max_Lat
    #% 
#%%    
# =============================================================================
# Function to calculat the distance between two points with given lat and lon
# =============================================================================
# The Haversine formula is needed for a correct calculation of the distance between points on the globe
    
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295  # Math.PI / 180
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p)*cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2
    return 12742 * asin(sqrt(a))  #2 * R; R = 6371 km
#%%
def closest_index(list_coord, point_coord):
    min_dist=min(list_coord, key=lambda point: distance(point_coord['Lat'],point_coord['Lon'],point['Lat'],point['Lon']))
    return list_coord.index(min_dist)
#
#def closest_xy(list_coord, point_coord):
#    min_dist=min(list_coord, key=lambda point: distance(point_coord['Lat'],point_coord['Lon'],point['Lat'],point['Lon']))
#    return list_coord.index(min_dist)

#%%
def read_hrrr_var(hrrr_ds, var_name, y, x, date_format):       
        ds_precip=hrrr_ds[var_name].isel(y=y,x=x)           
        ds_precip=ds_precip.resample(time='1D').sum("time").reset_coords(drop=True)                            
        date_list=pd.to_datetime(ds_precip.time.values).strftime(date_format).values.tolist()            
        if var_name=='PRATE_surface':
            precp_array=3600*ds_precip.values.astype('float')
        else:
            precp_array=ds_precip.values.astype('float')       
        precp_array[precp_array < 0] =np.nan
        precip_list=precp_array.tolist()
            
        return date_list, precip_list


#%%
#list_coord = [{'lat': 39.7612992, 'lon': -86.1519681}, 
#                {'lat': 39.762241,  'lon': -86.158436 }, 
#                {'lat': 39.7622292, 'lon': -86.1578917}]
#
#point_coord = {'lat': 39.7622290, 'lon': -86.1519750}
#print(closest(tempDataList, v))

##%%
#import mputil
#def distance2(point1, point2):
#    return mputil.haversine_distance(point1, point2)
#%%
# =============================================================================
#     Fonction to filter the gauge stations, input is the serie of na analysis with stn name as index
# =============================================================================
def stn_filter(ivar,stn_info,limit_na,df_anal):
    #%
    stn_out=pd.DataFrame(columns=stn_info.columns)
    df_anal_out=pd.DataFrame(index=df_anal.index)
    for istn in df_anal.columns:
        if df_anal.loc[ivar,istn]<=limit_na:
            stn_out=stn_out.append(stn_info.loc[stn_info['Stn_id']==istn])
            df_anal_out=df_anal_out.join(df_anal.loc[:,istn])
#%   
    return stn_out, df_anal_out