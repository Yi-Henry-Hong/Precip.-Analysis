# Read and plotting netcdf data using xarray package
# Compare these data with the timeseries data of the rain gauge
# For the merged MPE data
# Edited by Yi Hong, Mars 3, 2020

#%% Load packages
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
#from pandas.plotting import register_matplotlib_converters
import os
import numpy as np
import datetime
#import re
import matplotlib.dates as mdates
from precip_functions import date_filename, Read_rain_file, resamp_precip, resamp_df, MAE_RMSE_Diff, PBIAS, R2, map_area, closest_index

os.environ["PROJ_LIB"] = "C:\\ProgramData\\Anaconda3\\Library\\share"; #path of the proj for basemap
from mpl_toolkits.basemap import Basemap

#% calculate the minimum distance
from heapq import nsmallest
from operator import itemgetter
import csv

#%% Read netcdf data, mrcc_mrg product is daily data, daily precip accumulation
precp_product="MPE"
product_path="C:\\Reseach_CIGLR\\Precipitation\\Precip_products\\MPE\\MPE"

#%%
test_nc=xr.open_dataset(os.path.join(product_path,'mpeGL_20100101.nc'))
test_nc2=xr.open_dataset(os.path.join(product_path,'mpeGL_20161126.nc'))
#%%

#%% check if all the nc files with the same dimensions
wrong_dim=pd.DataFrame(columns=['nc_file', 'dim'])
for filename in os.listdir(product_path):
    mpe_ds=xr.open_dataset(os.path.join(product_path, filename))       
    if mpe_ds.sizes['dim']!=94069:
       wrong_dim=wrong_dim.append({'nc_file':filename, 'dim': int(mpe_ds.sizes['dim'])}, ignore_index=True) 

wrong_dim.to_csv(os.path.join('C:\\Reseach_CIGLR\\Precipitation\\Precip_products\\MPE','Different_dim.csv'),sep=';')
    
#=============================================================================
# #%% First, calculate the index of the measured stations
    # Overlake stations and Overland stations
    # It can be calculated once at the beginning
#=============================================================================
#%% Overlake stations, the 2 ReCON stations
# The index in dim for srl and wsl stations, daily accumulated precip in mm
# coordinate of Spectacle Reef (Huron) and White Shoal (Michigan), in N - E
# There is only one dim, the index corresponding to srl and wsl
loc_srl={'Lat':45.77319997,'Lon':-84.13669}    # 11169 if using the Haversine formula
loc_wsl={'Lat':45.84221668,'Lon':-85.1352833}   # 10546 if using the Haversine formula
#%%
list_dict_coord = [{'Lat':test_nc.latitude[i], 'Lon':test_nc.longitude[i]} for i in range(test_nc.latitude.size)]
#%%
index_min_srl=closest_index(list_dict_coord,loc_srl)
index_min_wsl=closest_index(list_dict_coord,loc_wsl)
#%% results of the 2 recon stations
# when dim = 94069,93737
index_min_srl=52586 
index_min_wsl=50383   
# only two nc file dim=182094, 188138, not checked      

##################################3
#%% Index for the 1287 overland stations
dir_stn='C:\\Reseach_CIGLR\\Precipitation\\station_observations'
stn_info_file=os.path.join(dir_stn,'stations_since_2010.txt')
if not os.path.isfile(stn_info_file): raise Exception('Station information file not exist')
col_names=['Stn_id','Stn_name','Lon','Lat']
stn_info= pd.read_csv(stn_info_file, sep=r'\s\s\s+', header=None, skipinitialspace=True,engine='python', names=col_names)

dir_stn_data=os.path.join(dir_stn,'OL_stations')
if not os.path.isdir(dir_stn_data): raise Exception('Station data directory not exist')

mpe_stn_index=dict.fromkeys(stn_info['Stn_id'].astype(str).tolist(), None)
#%%
for istn in range(len(stn_info)):
    point_coord={'Lat':stn_info['Lat'].iloc[istn], 'Lon':stn_info['Lon'].iloc[istn]}    
    mpe_stn_index[str(stn_info['Stn_id'].iloc[istn])]=closest_index(list_dict_coord,point_coord)

#% write the index dictionary
index_filename="C:\\Reseach_CIGLR\\Precipitation\\Precip_products\\MPE\\mpe_stn_index.csv"

with open (index_filename, "w") as f:
    for k, v in mpe_stn_index.items():
        f.write(str(k) + ';'+ str(v) + '\n')

#%%
max(mpe_stn_index.values())      # 83963  
# =============================================================================
# For overlake stations
        # open netcdf and generate list of daily precip data
        # This part of the script was run once for generate the data files
# =============================================================================
#%%
out_dir_lake="C:\\Reseach_CIGLR\\Precipitation\\Precip_products\\MPE\\Precip_Lake"
gauge=['huron_srl','mich_wsl']
index_lake=pd.DataFrame({'Stn_id':gauge, 'dim':[index_min_srl,index_min_wsl]})
mpe_precip_lake=dict([(key, pd.DataFrame(columns=['Date','Precip'])) for key in gauge]) 

#%%
for igauge in gauge:
    for filename in os.listdir(product_path):
        str_date=date_filename(filename)
        if len(str_date)>0:
           mpe_ds=xr.open_dataset(os.path.join(product_path, filename))
           if mpe_ds.sizes['dim']<100000:
               precp_data=mpe_ds['prcp_mm'].isel(dim=index_lake['dim'].loc[index_lake['Stn_id']==igauge]).data
               precp_data=pd.to_numeric(precp_data,errors='coerce')
               if precp_data<0: precp_data=np.nan
           else: precp_data=np.nan
           mpe_precip_lake[igauge]=mpe_precip_lake[igauge].append({'Date':str_date, 'Precip': float(precp_data)}, ignore_index=True)

#%% sort and write down
date_format='%Y%m%d'
begin_date='20100101'
end_date='20191231'
years=list(range(2010,2020)) 
freq_in='D'        # gauge data
freq_out='D'           # desired daily data
min_count=1 
#%
nan_mpe=pd.DataFrame(index=gauge,columns=['NaN_in','NaN_out'])     # the annual daily NA for each station
#%%  NaN analysis and write data
for igauge in gauge:
    df_precip=mpe_precip_lake[igauge]
    df_precip['Date']=pd.to_datetime(df_precip['Date'],format=date_format)
    df_precip=df_precip.sort_values(by=['Date']).set_index('Date')
    precip_query, na_in, na_out=resamp_df(df_precip, freq_in, freq_out, min_count, begin_date, end_date, date_format)
    
    nan_mpe.loc[igauge,'NaN_in']=na_in
    nan_mpe.loc[igauge,'NaN_out']=na_out    
    precip_query.to_csv(os.path.join(out_dir_lake,'MPE_Day_'+igauge+'_'+str(years[0])+'_'+str(years[-1])+'.csv'),sep=';')
    
nan_mpe.to_csv(os.path.join(out_dir_lake,'NA_OL_MPE_'+str(years[0])+'_'+str(years[-1])+'.csv'),sep=';')
  
#%%
# =============================================================================
# Compare the MPE overlake with the gauge overlake, year by year, using recorded data files       
# =============================================================================
#%% plot pictures of daily results and calculate the MAE and RMSE
dir_produc=out_dir_lake
dir_obs='C:\\Reseach_CIGLR\\Precipitation\\station_observations\\ReCON_stations'

years_prod=list(range(2010,2020))
years_obs=list(range(2013,2020))

index_anal=[str(iyear) for iyear in years]
index_anal.append('Total')
dict_anal=dict([(key, pd.DataFrame(index=index_anal, columns=gauge)) for key in ['MAE','RMSE','Anual_diff']]) 

#%%
for igage in gauge:
    rain_obs=pd.read_csv(os.path.join(dir_obs,'Precip_Day_'+str(igage)+'_'+str(years_obs[0])+'_'+str(years_obs[-1])+'.csv'),delimiter=';',names=['Date','Precip'],skiprows=1)
    rain_prod=pd.read_csv(os.path.join(dir_produc,'MPE_Day_'+str(igage)+'_'+str(years_prod[0])+'_'+str(years_prod[-1])+'.csv'),delimiter=';',names=['Date','Precip'],skiprows=1)

    for iyear in years_obs:
    #% creat output directory
        out_dir=os.path.join(dir_produc,str(iyear))
        if not os.path.isdir(out_dir):  
            os.mkdir(out_dir)

        begin_date=str(iyear)+'-01-01'
        end_date=str(iyear)+'-12-31' 
        obs_compare=rain_obs.loc[(pd.to_datetime(rain_obs['Date'])>=pd.to_datetime(begin_date)) & (pd.to_datetime(rain_obs['Date'])<=pd.to_datetime(end_date))].drop_duplicates()
        prod_compare=rain_prod.loc[(pd.to_datetime(rain_prod['Date'])>=pd.to_datetime(begin_date)) & (pd.to_datetime(rain_prod['Date'])<=pd.to_datetime(end_date))].drop_duplicates()
        #% prod_compare.dtypes
        obs_compare=obs_compare.set_index('Date')
        prod_compare=prod_compare.set_index('Date')
        obs_compare.index = pd.to_datetime(obs_compare.index)
        prod_compare.index = pd.to_datetime(prod_compare.index)
        #% MAE, RMSE
        MAE, RMSE=MAE_RMSE(obs_compare,prod_compare)
        if float(obs_compare.sum())>0:
            anual_diff=(prod_compare.sum()-obs_compare.sum())/obs_compare.sum()
        else: anual_diff=np.nan
        #% plot
        ax=prod_compare.plot(linestyle='-',color='b',title=str(igage)+'-Daily-Precipitation-'+str(iyear))
        obs_compare.plot(ax=ax,linestyle=':',color='r',lw=4)
        ax.set(xlabel='Date',ylabel='Precipitation (mm/day)')
        ax.legend(labels=[precp_product,'Observation'],loc='upper right')
        textstr = 'MAE=%.3f\nRMSE=%.3f\n(Mrg-Obs)/Obs=%.3f\n'%(MAE,RMSE,anual_diff)
        ax.annotate(textstr, xy=(0.01, 0.8), xycoords="axes fraction")
        #%
        #ax.figure.autofmt_xdate()
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.tick_params(axis='x', rotation=70)
        #plt.show()
        fig = ax.get_figure()
        fig.savefig(os.path.join(out_dir,str(igage)+'_daily_'+precp_product+'.jpg'))
        
        dict_anal['MAE'].loc[str(iyear),igage]=float(MAE)
        dict_anal['RMSE'].loc[str(iyear),igage]=float(RMSE)
        dict_anal['Anual_diff'].loc[str(iyear),igage]=float(anual_diff)
#% Total
    total_begin_date=str(years_obs[0])+'0101'
    total_end_date=str(years_obs[-1])+'1231' 
    total_obs_compare=rain_obs.loc[(pd.to_datetime(rain_obs['Date'])>=pd.to_datetime(total_begin_date)) & (pd.to_datetime(rain_obs['Date'])<=pd.to_datetime(total_end_date))].drop_duplicates()
    total_prod_compare=rain_prod.loc[(pd.to_datetime(rain_prod['Date'])>=pd.to_datetime(total_begin_date)) & (pd.to_datetime(rain_prod['Date'])<=pd.to_datetime(total_end_date))].drop_duplicates()
    #% prod_compare.dtypes
    total_obs_compare=total_obs_compare['Precip']
    total_prod_compare=total_prod_compare['Precip']
    #% MAE, RMSE, diff
    total_MAE, total_RMSE=MAE_RMSE(total_obs_compare,total_prod_compare)
    if float(total_obs_compare.sum())>0:       
        total_anual_diff=(total_prod_compare.sum()-total_obs_compare.sum())/total_obs_compare.sum()
    else: total_anual_diff=np.nan
    #%
    dict_anal['MAE'].loc['Total',igage]=float(total_MAE)        
    dict_anal['RMSE'].loc['Total',igage]=float(total_RMSE)         
    dict_anal['Anual_diff'].loc['Total',igage]=float(total_anual_diff)    # precipitation annual difference 
 
#%
dict_anal['MAE'].to_csv(os.path.join(dir_produc,'MAE_Lake_MPE_'+str(years[0])+'_'+str(years[-1])+'.csv'),sep=';')
dict_anal['RMSE'].to_csv(os.path.join(dir_produc,'RMSE_Lake_MPE_'+str(years[0])+'_'+str(years[-1])+'.csv'),sep=';')
dict_anal['Anual_diff'].to_csv(os.path.join(dir_produc,'Anual_diff_Lake_MPE_'+str(years[0])+'_'+str(years[-1])+'.csv'),sep=';')
        
#%%
############################################################################################################
# =============================================================================#############################
# For overland stations       
# =============================================================================##############################
##############################################################################################################
out_dir_land="C:\\Reseach_CIGLR\\Precipitation\\Precip_products\\MPE\\Precip_Land"
# index and dataframe of analysis results
index_file="C:\\Reseach_CIGLR\\Precipitation\\Precip_products\\MPE\\mpe_stn_index.csv"
mpe_index=pd.read_csv(index_file, sep=';', header=None, names=['Stn_id','dim'])
mpe_precip=dict([(key, pd.DataFrame(columns=['Date','Precip'])) for key in mpe_index['Stn_id'].astype(str).tolist()]) 

#%%
for idim in range(len(mpe_index)):
    stn_name=str(mpe_index['Stn_id'].iloc[idim])
    for filename in os.listdir(product_path):
        str_date=date_filename(filename)
        if len(str_date)>0:
           mpe_ds=xr.open_dataset(os.path.join(product_path, filename))
           if mpe_ds.sizes['dim']<100000:
               precp_data=mpe_ds['prcp_mm'].isel(dim=mpe_index['dim'].iloc[idim]).data
               precp_data=pd.to_numeric(precp_data,errors='coerce')
               if precp_data<0: precp_data=np.nan
           else: precp_data=np.nan                             
           mpe_precip[stn_name]=mpe_precip[stn_name].append({'Date':str_date, 'Precip': float(precp_data)}, ignore_index=True)

#% sort and write down
date_format='%Y%m%d'
begin_date='20100101'
end_date='20191231'
years=list(range(2010,2020)) 
freq_in='D'        # gauge data
freq_out='D'           # desired daily data
min_count=1 
#%
nan_mpe=pd.DataFrame(index=mpe_index['Stn_id'].astype(str).tolist(),columns=['NaN_in','NaN_out'])     # the annual daily NA for each station
#%  NaN analysis and write data
for precip_df in mpe_precip:
    df_precip=mpe_precip[precip_df]
    df_precip['Date']=pd.to_datetime(df_precip['Date'],format=date_format)
    df_precip=df_precip.sort_values(by=['Date']).set_index('Date')

    precip_query, na_in, na_out=resamp_df(df_precip, freq_in, freq_out, min_count, begin_date, end_date, date_format)
    
    nan_mpe.loc[precip_df,'NaN_in']=na_in
    nan_mpe.loc[precip_df,'NaN_out']=na_out    
    precip_query.to_csv(os.path.join(out_dir_land,'Data\\MPE_Day_'+str(precip_df)+'_'+str(years[0])+'_'+str(years[-1])+'.csv'),sep=';')
    
nan_mpe.to_csv(os.path.join(out_dir_land,'NA_Land_mpe_'+str(years[0])+'_'+str(years[-1])+'.csv'),sep=';')
##############################################################################################
#%% Calculate RMSE and MAE, PBias, R2
###################################################################################################
out_dir_land="C:\\Reseach_CIGLR\\Precipitation\\Precip_products\\MPE\\Precip_Land"

years=list(range(2010,2020))
index_anal=[str(iyear) for iyear in years]
index_anal.append('Total')
col_names=stn_info['Stn_id'].astype(str).tolist()

df_MAE=pd.DataFrame(index=index_anal,columns=col_names)
df_RMSE=pd.DataFrame(index=index_anal,columns=col_names)
df_anual_diff=pd.DataFrame(index=index_anal,columns=col_names)
df_PBias=pd.DataFrame(index=index_anal,columns=col_names)
df_r2=pd.DataFrame(index=index_anal,columns=col_names)

dir_obs='C:\\Reseach_CIGLR\\Precipitation\\station_observations\\OL_stn2'
#%%  
for stn in col_names:
    #%
    rain_obs=pd.read_csv(os.path.join(dir_obs,'Precip_Day_'+str(stn)+'_'+str(years[0])+'_'+str(years[-1])+'.csv'),delimiter=';',names=['Date','Precip'],skiprows=1)
    rain_prod=pd.read_csv(os.path.join(out_dir_land,'Data\\MPE_Day_'+str(stn)+'_'+str(years[0])+'_'+str(years[-1])+'.csv'),delimiter=';',names=['Date','Precip'],skiprows=1)
#%
    for iyear in years:
        #%
        begin_date=str(iyear)+'-01-01'
        end_date=str(iyear)+'-12-31' 
        obs_compare=rain_obs.loc[(pd.to_datetime(rain_obs['Date'])>=pd.to_datetime(begin_date)) & (pd.to_datetime(rain_obs['Date'])<=pd.to_datetime(end_date))].drop_duplicates()
        prod_compare=rain_prod.loc[(pd.to_datetime(rain_prod['Date'])>=pd.to_datetime(begin_date)) & (pd.to_datetime(rain_prod['Date'])<=pd.to_datetime(end_date))].drop_duplicates()
        #% prod_compare.dtypes
        obs_compare=obs_compare['Precip'].astype(float)
        prod_compare=prod_compare['Precip'].astype(float)        
        #% Delete outliers
        for i in range(len(obs_compare)):
            if obs_compare.iloc[i]<0 or obs_compare.iloc[i]>150:
                obs_compare.iloc[i]=np.nan
        for i in range(len(prod_compare)):
            if prod_compare.iloc[i]<0 or prod_compare.iloc[i]>150:
                prod_compare.iloc[i]=np.nan  
        #% MAE, RMSE
        MAE, RMSE, Diff=MAE_RMSE_Diff(prod_compare, obs_compare)
        PBias=PBIAS(prod_compare, obs_compare)
        r2=R2(prod_compare, obs_compare)
        #% write to dataframes
        df_MAE.loc[str(iyear),stn]=float(MAE)        
        df_RMSE.loc[str(iyear),stn]=float(RMSE)         
        df_anual_diff.loc[str(iyear),stn]=float(Diff)    # precipitation annual difference 
        df_PBias.loc[str(iyear),stn]=float(PBias) 
        df_r2.loc[str(iyear),stn]=float(r2) 
        
    #% calculate the total difference     
    total_begin_date=str(years[0])+'0101'
    total_end_date=str(years[-1])+'1231' 
    total_obs_compare=rain_obs.loc[(pd.to_datetime(rain_obs['Date'])>=pd.to_datetime(total_begin_date)) & (pd.to_datetime(rain_obs['Date'])<=pd.to_datetime(total_end_date))].drop_duplicates()
    total_prod_compare=rain_prod.loc[(pd.to_datetime(rain_prod['Date'])>=pd.to_datetime(total_begin_date)) & (pd.to_datetime(rain_prod['Date'])<=pd.to_datetime(total_end_date))].drop_duplicates()
    #% prod_compare.dtypes
    total_obs_compare=total_obs_compare['Precip']
    total_prod_compare=total_prod_compare['Precip']
    #% Delete outliers
    for i in range(len(total_obs_compare)):
        if total_obs_compare.iloc[i]<0 or total_obs_compare.iloc[i]>150:
            total_obs_compare.iloc[i]=np.nan
    for i in range(len(total_prod_compare)):
        if total_prod_compare.iloc[i]<0 or total_prod_compare.iloc[i]>150:
            total_prod_compare.iloc[i]=np.nan  

    #% MAE, RMSE, diff
    total_MAE, total_RMSE, total_Diff=MAE_RMSE_Diff(total_prod_compare, total_obs_compare)
    total_PBias=PBIAS(total_prod_compare, total_obs_compare)
    total_r2=R2(total_prod_compare, total_obs_compare)
    #%
    df_MAE.loc['Total',stn]=float(total_MAE)        
    df_RMSE.loc['Total',stn]=float(total_RMSE)         
    df_anual_diff.loc['Total',stn]=float(total_Diff)/len(years)   # precipitation annual difference 
    df_PBias.loc['Total',stn]=float(total_PBias)
    df_r2.loc['Total',stn]=float(total_r2) 
#%        
df_MAE.to_csv(os.path.join(out_dir_land,'MAE_OL_MPE_'+str(years[0])+'_'+str(years[-1])+'.csv'),sep=';')
df_RMSE.to_csv(os.path.join(out_dir_land,'RMSE_OL_MPE_'+str(years[0])+'_'+str(years[-1])+'.csv'),sep=';')
df_anual_diff.to_csv(os.path.join(out_dir_land,'Diff_OL_MPE_'+str(years[0])+'_'+str(years[-1])+'.csv'),sep=';')    
df_PBias.to_csv(os.path.join(out_dir_land,'PBias_OL_MPE_'+str(years[0])+'_'+str(years[-1])+'.csv'),sep=';')       
df_r2.to_csv(os.path.join(out_dir_land,'R2_OL_MPE_'+str(years[0])+'_'+str(years[-1])+'.csv'),sep=';')       

#%% plot maps, https://matplotlib.org/3.1.1/tutorials/colors/colormaps.html for color bar options
dir_stn='C:\\Reseach_CIGLR\\Precipitation\\station_observations'
stn_info_file=os.path.join(dir_stn,'anal_2010_2019\\stations_NaN_Fraction_0.1.csv')
col_names=['Stn_id','Stn_name','Lon','Lat']
stn_info= pd.read_csv(stn_info_file, sep=';', header=None, skiprows=1, skipinitialspace=True,engine='python', names=col_names)

shorline_file=os.path.join(dir_stn,'greatlakes.shoreline.dat2')
min_Lon,max_Lon,min_Lat,max_Lat=map_area(shorline_file)

#%% read csv data
out_dir_land="C:\\Reseach_CIGLR\\Precipitation\\Precip_products\\MPE\\Precip_Land"
df_MAE=pd.read_csv(os.path.join(out_dir_land,'MAE_OL_MPE_'+str(years[0])+'_'+str(years[-1])+'.csv'),sep=';',index_col=0)
df_RMSE=pd.read_csv(os.path.join(out_dir_land,'RMSE_OL_MPE_'+str(years[0])+'_'+str(years[-1])+'.csv'),sep=';',index_col=0)
df_anual=pd.read_csv(os.path.join(out_dir_land,'Diff_OL_MPE_'+str(years[0])+'_'+str(years[-1])+'.csv'),sep=';',index_col=0)    
df_PBias=pd.read_csv(os.path.join(out_dir_land,'PBias_OL_MPE_'+str(years[0])+'_'+str(years[-1])+'.csv'),sep=';',index_col=0)       
df_r2=pd.read_csv(os.path.join(out_dir_land,'R2_OL_MPE_'+str(years[0])+'_'+str(years[-1])+'.csv'),sep=';',index_col=0)       

#%%
list_var=['MAE','RMSE','anual_diff','PBias','r2']
v_min_max={'MAE':[0,5],'RMSE':[0,10],'anual_diff':[-100,100],'PBias':[-25,25],'r2':[0,1]}
#%%
for ivar in list_var:
    #%%
    ivar='PBias'
    for iindex in index_anal:
        #%
        df_anal=eval('df_'+ivar).copy()        
        df_anal=df_anal.loc[:,stn_info['Stn_id']]

        map = Basemap(projection='merc',resolution='i',lat_0=45,lon_0=-83, area_thresh=1000.0, llcrnrlon=min_Lon-2,llcrnrlat=min_Lat-2,urcrnrlon=max_Lon+2,urcrnrlat=max_Lat+2) #lat_0=45,lon_0=-83,
        parallels = np.arange(0.,81,5.)
        map.drawparallels(parallels,labels=[True,False,False,False],linewidth=0.1)  ## fontsize=ftz,
        meridians = np.arange(10.,351.,5.)
        map.drawmeridians(meridians,labels=[False,False,False,True],dashes=[8,1],linewidth=0.1)  ##fontsize=ftz,
        map.drawcoastlines() 
        
        if ivar in ['MAE','RMSE']:
            color_bar='RdYlGn_r'   #jet
        elif ivar in ['PBias','anual_diff']:
            color_bar='nipy_spectral'
        else:
            color_bar='RdYlGn'  #jet_r
                
        #% add points in the figure
        x,y=map(list(stn_info['Lon']),list(stn_info['Lat'])) 
        scatter = map.scatter(x,y,s=10,c=df_anal.loc[iindex], vmin=v_min_max[ivar][0], vmax=v_min_max[ivar][1], cmap=color_bar)
        cbar=plt.colorbar()
        
        if ivar in ['MAE','RMSE']:
            text_title= ivar+' for the year '+ str(iindex)        
            cbar.set_label(ivar+' (mm/day)',labelpad=10, y=0.45, rotation=270)       
        elif ivar=='anual_diff':
            text_title= 'Accumulated difference for the year of '+ str(iindex)
            cbar.set_label('Prod.-Obs. (mm)',labelpad=10, y=0.45, rotation=270)    
        elif ivar=='PBias':
            text_title= 'PBias for the year of '+ str(iindex)
            cbar.set_label('PBias (%)',labelpad=10, y=0.45, rotation=270)
        else:
            text_title= 'R2 for the year of '+ str(iindex)
            cbar.set_label('R2 ',labelpad=10, y=0.45, rotation=270)
        
        
        plt.title(text_title)
        plt.savefig(os.path.join('C:\\Reseach_CIGLR\\Precipitation\\Precip_products\\MPE','Figures\\Annual_Figs\\'+str(iindex)+'_Map_'+ivar+'.jpg'))
        plt.show()           

#%% boxplot of each var
ylim={'MAE':[0,10],'RMSE':[0,20],'anual_diff':[-500, 500],'PBias':[-100,100],'r2':[0,1]}
for ivar in list_var:       
    df_anal=eval('df_'+ivar).astype('float')
    if ivar in ['MAE','RMSE']:
        text_title= 'Boxplots of '+ivar        
    elif ivar=='anual_diff':
        text_title= 'Boxplots of accumulated difference'   
    elif ivar=='r2':
        text_title= 'Boxplots of R2'     
    else:
        text_title= 'Boxplots of PBIAS'
    
    ax = df_anal.T.boxplot() 
    ax.set_ylim(ylim[ivar])
    plt.title(text_title)
    plt.savefig(os.path.join('C:\\Reseach_CIGLR\\Precipitation\\Precip_products\\MPE','Figures\\Boxplots_'+ivar+'.jpg'))
    plt.show()  
   