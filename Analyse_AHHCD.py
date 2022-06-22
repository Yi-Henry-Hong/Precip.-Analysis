# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
# ** This script is for evaluating the the performance of different products in US and CAN
# ** Add calculations of AHHCD product
# ** Edit by Yi Hong, 10/09/2021
# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

#%%    
# =============================================================================
# Import modules
# =============================================================================
#import sys
import os
import datetime
import numpy as np
#import xarray as xr

import pandas as pd
import matplotlib.pyplot as plt

import geopandas as gpd
from precip_functions import MAE_RMSE_Diff, PBIAS, R2,resamp_df
import re

#os.environ["PROJ_LIB"] = "C:\\ProgramData\\Anaconda3\\Library\\share"; #path of the proj for basemap
#from mpl_toolkits.basemap import Basemap

import seaborn as sns

#%%
# =============================================================================
# Configuration settings
# =============================================================================
precp_product=['AORC','CaPA','MPE','Mrg']
# precp_product=['AORC','AORC10','CaPA','MPE','MPE10','Mrg']

dir_stn='C:\\Research_Mich\\Precipitation\\station_observations'
dir_stn_data=os.path.join(dir_stn,'OL_stn2')
limit_na=0.1
stn_info_file=os.path.join(dir_stn,'anal_2010_2019\\stations_NaN_Fraction_'+str(limit_na)+'.csv')
stn_info= pd.read_csv(stn_info_file, sep=';',skipinitialspace=True)   #col_names=['Stn_id','Stn_name','Lon','Lat']


#%%
# =============================================================================
# Comparing with station locations of GHCN - AHHCD
# =============================================================================
#%%
dir_ahhcd=r"C:\Research_Mich\Precipitation\station_observations\AHCCD"
col_names=['Prov','station name','stnid','beg_yr','beg_mon','end_yr','end_mon','lat_deg','lon_deg','elev_m','stns_joined']
stn_ahhcd=pd.read_csv(os.path.join(dir_ahhcd,'AHHCD_stn.csv'),header=None, names=col_names, sep=',')  # A total of 463 stations
#rng_date2 = pd.date_range(begin_time2,end_time2,freq='D')
valid_stn=stn_ahhcd.loc[stn_ahhcd['end_yr']==2017]  # only 89 stations contain records until 2017

#%%
# =============================================================================
# Points within Canada, and plot
# =============================================================================
ShpDir_country =r"C:\Research_Mich\Precipitation\shapefiles\ne_50m_admin_0_countries"      # Catchment directory
in_shp_wgs84 = gpd.read_file(os.path.join(ShpDir_country,'ne_50m_admin_0_countries.shp'))
shp_US=in_shp_wgs84.loc[in_shp_wgs84['NAME_EN']=='United States of America']
shp_CA=in_shp_wgs84.loc[in_shp_wgs84['NAME_EN']=='Canada']

#%
lon = valid_stn['lon_deg']
lat = valid_stn['lat_deg']
pnts= gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat))
#%%
fig, ax = plt.subplots()
shp_US.plot(ax=ax, facecolor='lightgrey')
shp_CA.plot(ax=ax, facecolor='bisque')
pnts.plot(ax=ax, color='red', markersize=3)
ax.set_xlim(-150, -50)
ax.set_ylim(30, 75) 
plt.title('89 valid AHCCD stations (records until 2017) in Canada')
#ax.legend(loc='upper right')
plt.savefig(os.path.join(dir_ahhcd,'Station_locations.jpg'))
plt.show()

#%% 
# =============================================================================
# mask for 89 AHHCD stations within the Great Lakes region
# =============================================================================
ShpDir =r"C:\Research_Mich\Precipitation\shapefiles"      # Catchment directory
in_basin_wgs84 = gpd.read_file(os.path.join(ShpDir,r'shp_wgs84\basin_wgs84.shp'))
in_lake_wgs84 = gpd.read_file(os.path.join(ShpDir,r'shp_wgs84\lake_wgs84.shp'))
cut_lake = gpd.read_file(os.path.join(ShpDir,r'shp_wgs84\land_wgs84.shp'))
#%
fig, ax = plt.subplots()
in_basin_wgs84.plot(ax=ax, facecolor='silver')
in_lake_wgs84.plot(ax=ax, facecolor='skyblue')
pnts.plot(ax=ax, color='red', markersize=4) 
ax.set_xlim(-95, -74)
ax.set_ylim(40, 52)  
plt.title('AHCCD station locations in Great Lake basins')
plt.savefig(os.path.join(dir_ahhcd,'Station_locations_GreatLakes.jpg'))
plt.show()

#%% The entire Great Lakes region
basin=in_basin_wgs84.loc[in_basin_wgs84['LAKE']=='Union']
basin.reset_index(drop=True, inplace=True)
mask_basin=pnts.within(basin.loc[0,'geometry'])

sum(mask_basin)  # check, 7 stations in total
#% apply the mask to valid_stn
valid_stn.reset_index(drop=True, inplace=True)
stn_GL=valid_stn[mask_basin]
stn_GL.reset_index(drop=True, inplace=True)
stn_GL.to_csv(os.path.join(dir_ahhcd,'Stations_GreatLakes.csv'), sep=',', index=False) 

#%%
# =============================================================================
# # Check if there are any repetition of the GHCN-D stations
# =============================================================================
stn_equal=pd.DataFrame(index=range(7),columns=stn_info.columns.to_list()+['stnid_AHHCD','stn_name_AHHCD','Lon_AHHCD','Lat_AHHCD'])

new_index=0
for i in range(stn_info.shape[0]):    
    for j in range (stn_GL.shape[0]):
        dis_lon=abs(stn_info.loc[i,'Lon']-stn_GL.loc[j,'lon_deg'])
        dis_lat=abs(stn_info.loc[i,'Lat']-stn_GL.loc[j,'lat_deg'])        
        if (dis_lon<0.1 and dis_lat<0.1):
            stn_equal.loc[new_index,'Stn_id'] = stn_info.loc[i,'Stn_id']
            stn_equal.loc[new_index,'Stn_name'] = stn_info.loc[i,'Stn_name']
            stn_equal.loc[new_index,'Lon'] = stn_info.loc[i,'Lon']  
            stn_equal.loc[new_index,'Lat'] = stn_info.loc[i,'Lat']  
            
            stn_equal.loc[new_index,'stnid_AHHCD'] = stn_GL.loc[j,'stnid']      
            stn_equal.loc[new_index,'stn_name_AHHCD'] = stn_GL.loc[j,'station name']                   
            stn_equal.loc[new_index,'Lon_AHHCD'] = stn_GL.loc[j,'lon_deg']  
            stn_equal.loc[new_index,'Lat_AHHCD'] = stn_GL.loc[j,'lat_deg'] 
            new_index = new_index+1

stn_equal.drop(stn_equal.index[[0,2,3,5]],inplace=True)  # mannually delete repeated stations

stn_equal.to_csv(os.path.join(dir_ahhcd,'Stations_equal.csv'), sep=',', index=False) 

#%%
# =============================================================================
# Now, run analysis and compare for these 4 equal stations
# =============================================================================

begin_date='2010/01/01 00:00'
end_date='2017/12/31 00:00' 
date_format='%Y/%m/%d %H:%M'    

begin_time=datetime.datetime.strptime(begin_date, date_format)
end_time=datetime.datetime.strptime(end_date, '%Y/%m/%d %H:%M')
stn_equal['stnid_AHHCD'] = stn_equal['stnid_AHHCD'].astype(str)

freq_in='D'        # gauge data
freq_out='D'           # desired daily data
min_count=1   

df_anal=pd.DataFrame(index=stn_equal['stnid_AHHCD'].to_list(),columns=['Init_NA','Fra_Init_NA','Resamp_Day_NA','Fra_Resamp_NA'])
 
for new_id in stn_equal['stnid_AHHCD'].to_list():
    #%
    filename="dt"+new_id.strip()+'.txt'
    f=open(os.path.join(dir_ahhcd, 'Adj_Daily_Total_v2017\\'+filename), 'r', errors=None)
    record_file=f.readlines()
    f.close()
    #%
    # head_line=record_file[0].split(',')      
    precip_gauge=[]
   
    for line in record_file[1:-1]:
        #%
        line_txt=re.split('[A-Z]|\s+',line)   ## split the line, use regular expression, see the doc for data format      
        line_data = list(filter(None, line_txt))
        line_time=datetime.datetime.strptime(str(line_data[0]+'-'+line_data[1]),'%Y-%m')
        
        if line_time >= begin_time:
            for i in range(1,32):
                if float(line_data[i+1])>=0:
                    line_list=[]
                    line_list.append(line_data[0]+'-'+line_data[1]+'-'+str(i))
                    line_list.append(line_data[i+1])
                      
                    precip_gauge.append(line_list)   
#%
    df_precip = pd.DataFrame(precip_gauge, columns = ['Date', 'Precip'])
    df_precip['Date']=pd.to_datetime(df_precip['Date'],format='%Y-%m-%d')
    df_precip=df_precip.set_index('Date') 
    
    precip,na_init,na_resamp=resamp_df(df_precip,freq_in, freq_out,  min_count, begin_date, end_date, date_format)
    df_anal.loc[new_id,'Init_NA']=na_init
    df_anal.loc[new_id,'Fra_Init_NA']=na_init/(365*7)
    df_anal.loc[new_id,'Resamp_Day_NA']=na_resamp
    df_anal.loc[new_id,'Fra_Resamp_NA']=na_resamp/(365*7) 

    precip.to_csv(os.path.join(dir_ahhcd,'Precip_Day_'+str(new_id.strip())+'_2010_2017.csv'),sep=',')

df_anal.to_csv(os.path.join(dir_ahhcd,'NA_AHHCD_2010_2017.csv'),sep=',')

#%%
# =============================================================================
# After the check of NA data, only 2 stations is good for comparison,6163171 and 6150689
# =============================================================================
stn_equal.reset_index()
stn_equal.drop(stn_equal.index[[0,1]],inplace=True) # mannually drop
#%%
list_var=['MAE','RMSE','Diff','PBias','r2']
mul_index=pd.MultiIndex.from_product([stn_equal['Stn_id'],list_var, precp_product], names=['Station','Variables','Product'])

df_anal=pd.DataFrame(index=mul_index, columns = ['GHCN','AHCCD'])
#%%
for n_stn in range(2):  # only two stations
#%
    istn=stn_equal.iloc[n_stn,:]['Stn_id']
    stn_adj=stn_equal.iloc[n_stn,:]['stnid_AHHCD'].split()[0]
    
    rain_GHCN=pd.read_csv(os.path.join(dir_stn_data,'Precip_Day_'+istn+'_2010_2019.csv'),delimiter=';',names=['Date','Precip'],skiprows=1)
    rain_GHCN['Date']=pd.to_datetime(rain_GHCN['Date'])
    GHCN_comp=rain_GHCN.loc[(rain_GHCN['Date']>=begin_time) & (rain_GHCN['Date']<=end_time)].reset_index()['Precip']
#%   
    rain_AHHCD=pd.read_csv(os.path.join(dir_ahhcd,'Precip_Day_'+stn_adj+'_2010_2017.csv'),delimiter=',',names=['Date','Precip'],skiprows=1)
    rain_AHHCD['Date']=pd.to_datetime(rain_AHHCD['Date'])
    AHHCD_comp=rain_AHHCD.loc[(rain_AHHCD['Date']>=begin_time) & (rain_AHHCD['Date']<=end_time)].reset_index()['Precip']
#%
    for iprod in precp_product:
        #%
        dir_prod="C:\\Research_Mich\\Precipitation\\Precip_products\\"+iprod+"\\Precip_Land\\Data"
                
        if iprod == 'AORC10':
            rain_prod=pd.read_csv(os.path.join(dir_prod, 'AORC_Day_'+str(istn)+'_2010_2019.csv'),delimiter=';',names=['Date','Precip'],skiprows=1,index_col=0)
            rain_prod['Date']=pd.to_datetime(rain_prod['Date'],format='%Y%m%d')
        elif iprod == 'MPE10':
            rain_prod=pd.read_csv(os.path.join(dir_prod, 'MPE_Day_'+str(istn)+'_2010_2019.csv'),delimiter=';',names=['Date','Precip'],skiprows=1,index_col=0)
            rain_prod['Date']=pd.to_datetime(rain_prod['Date'],format='%Y%m%d')
        else:
            rain_prod=pd.read_csv(os.path.join(dir_prod, iprod+'_Day_'+str(istn)+'_2010_2019.csv'),delimiter=';',names=['Date','Precip'],skiprows=1)   
            rain_prod['Date']=pd.to_datetime(rain_prod['Date'])

        #%                                
        if (rain_prod['Date'].iloc[0]<=begin_time) & (rain_prod['Date'].iloc[-1]>=end_time):
            prod_comp=rain_prod.loc[(rain_prod['Date']>=begin_time) & (rain_prod['Date']<=end_time)].reset_index()['Precip']
        else:
            compare_date = pd.date_range(begin_time,end_time,freq='D')
            df_compare=pd.DataFrame(index=compare_date, columns=['Precip'])                    
            temp_prod=rain_prod[rain_prod['Date'].isin(compare_date)]
            #%
            if len(temp_prod)>0:
                for i in range(len(temp_prod)):
                    df_compare.loc[temp_prod['Date'].iloc[i], 'Precip']=temp_prod['Precip'].iloc[i]
            prod_comp=df_compare['Precip']              
        #% Delete outliers
        for i in range(len(GHCN_comp)):
            if GHCN_comp.iloc[i]<0 or GHCN_comp.iloc[i]>150:
                GHCN_comp.iloc[i]=np.nan
        for i in range(len(AHHCD_comp)):
            if AHHCD_comp.iloc[i]<0 or AHHCD_comp.iloc[i]>150:
                AHHCD_comp.iloc[i]=np.nan
        for i in range(len(prod_comp)):
            if prod_comp.iloc[i]<0 or prod_comp.iloc[i]>150:
                prod_comp.iloc[i]=np.nan  
                
        #% MAE, RMSE, diff, PBias, r2
        MAE1, RMSE1, Diff1=MAE_RMSE_Diff(prod_comp, GHCN_comp)
        PBias1=PBIAS(prod_comp, GHCN_comp)
        r2_G=R2(prod_comp, GHCN_comp)
        
        MAE2, RMSE2, Diff2=MAE_RMSE_Diff(prod_comp, AHHCD_comp)
        PBias2=PBIAS(prod_comp, AHHCD_comp)
        r2_A=R2(prod_comp, AHHCD_comp)        
    
        df_anal.loc[(istn,'MAE',iprod), 'GHCN'] = float(MAE1) 
        df_anal.loc[(istn,'RMSE',iprod), 'GHCN'] = float(RMSE1)  
        df_anal.loc[(istn,'Diff',iprod), 'GHCN'] = float(Diff1)/8
        df_anal.loc[(istn,'PBias',iprod), 'GHCN'] = float(PBias1)
        df_anal.loc[(istn,'r2',iprod), 'GHCN'] = float(r2_G)

        df_anal.loc[(istn,'MAE',iprod), 'AHCCD'] = float(MAE2) 
        df_anal.loc[(istn,'RMSE',iprod), 'AHCCD'] = float(RMSE2)  
        df_anal.loc[(istn,'Diff',iprod), 'AHCCD'] = float(Diff2)/8
        df_anal.loc[(istn,'PBias',iprod), 'AHCCD'] = float(PBias2)
        df_anal.loc[(istn,'r2',iprod), 'AHCCD'] = float(r2_A)
#%%
df1 = df_anal.unstack(level=-1).unstack(level=-1)
df2=df1.swaplevel(1,axis=1).swaplevel(0,axis=1)
df2.sort_index(axis=1, level=0, inplace=True)
df2.to_csv(os.path.join(dir_ahhcd,'Anal_AHHCD_2010_2017.csv'),sep=',')


#%%
# =============================================================================
# Compare with adjusted canadian stations
# =============================================================================
#%%
for n_stn in range(2):  # only two stations
#%
    istn=stn_equal.iloc[n_stn,:]['Stn_id']
    stn_adj=stn_equal.iloc[n_stn,:]['stnid_AHHCD'].split()[0]
    
    rain_GHCN=pd.read_csv(os.path.join(dir_stn_data,'Precip_Day_'+istn+'_2010_2019.csv'),delimiter=';',names=['Date','Precip'],skiprows=1)
    rain_GHCN['Date']=pd.to_datetime(rain_GHCN['Date'])
    GHCN_comp=rain_GHCN.loc[(rain_GHCN['Date']>=begin_time) & (rain_GHCN['Date']<=end_time)].reset_index()['Precip']
#%   
    rain_AHHCD=pd.read_csv(os.path.join(dir_ahhcd,'Precip_Day_'+stn_adj+'_2010_2017.csv'),delimiter=',',names=['Date','Precip'],skiprows=1)
    rain_AHHCD['Date']=pd.to_datetime(rain_AHHCD['Date'])
    AHHCD_comp=rain_AHHCD.loc[(rain_AHHCD['Date']>=begin_time) & (rain_AHHCD['Date']<=end_time)].reset_index()['Precip']

#%
    precip_GHCN_mon=rain_GHCN.set_index('Date').copy()
    precip_AHCCD_mon=rain_AHHCD.set_index('Date').copy()

    precip_GHCN_mon=precip_GHCN_mon.loc[(precip_GHCN_mon.index>=begin_time) & (precip_GHCN_mon.index<=end_time)]
    precip_AHCCD_mon=precip_AHCCD_mon.loc[(precip_AHCCD_mon.index>=begin_time) & (precip_AHCCD_mon.index<=end_time)]

#%
    precip_GHCN_mon=precip_GHCN_mon.resample("M").sum()
    precip_AHCCD_mon=precip_AHCCD_mon.resample("M").sum()

#%
    fig,ax = plt.subplots()     
    precip_GHCN_mon.rename({'Precip': 'GHCN'}, axis=1, inplace=True)
    precip_GHCN_mon['GHCN'].plot()
    precip_AHCCD_mon.rename({'Precip': 'AHCCD'}, axis=1, inplace=True)
    precip_AHCCD_mon['AHCCD'].plot()
    # precip_AHCCD_mon(axis=1).plot()
    ax.set_title('Monthly precipitation at station '+stn_equal.iloc[n_stn,:]['stn_name_AHHCD'].split()[0],  fontsize=16, weight='bold')
    ax.legend(['GHCN','AHCCD'])  
    ax.legend(ncol=2, fontsize=14).set_title('')
    ax.set_ylabel('Precipitation (mm/Month)', fontsize=14, weight='bold')
    ax.set_xlabel('Date', fontsize=14, weight='bold')
    ax.tick_params(axis='both', labelsize=14)
    plt.savefig(os.path.join(dir_ahhcd,'GHCN_AHCCD_'+stn_adj+'.jpg'))



