# Read and plotting netcdf data using xarray package
# Indice file for the over-land stations aorc_stninfo_precip_ij.txt
# Edited by Yi Hong, Oct. 31 2019
# Modified by Yi Hong, Mars 18, 2020, to add 2019 data

# Load the xarray package
import xarray as xr     # version 0.13.0
import netCDF4 as nc4   #  version 1.5.2
import matplotlib.pyplot as plt
import pandas as pd  # version 0.25.2
#from pandas.plotting import register_matplotlib_converters
import os
import numpy as np   # version 1.15.4
import datetime
import matplotlib.dates as mdates
from precip_functions import AORC_raingauge, Read_rain_file, resamp_precip, MAE_RMSE_Diff, map_area,resamp_df

os.environ["PROJ_LIB"] = "C:\\ProgramData\\Anaconda3\\Library\\share"; #path of the proj for basemap
from mpl_toolkits.basemap import Basemap


#%% test
test_nc=xr.open_dataset('C:\\Reseach_CIGLR\\Precipitation\\Precip_products\\AORC\\2010010100.LDASIN_DOMAIN1')
test2=xr.open_dataset('D:\\AORC\\2012_regridded\\201204301300.PRECIP_FORCING.nc')

#%%
# =============================================================================
# For overlake stations
        # open netcdf and generate list of daily precip data
        # This part of the script was run once for generate the data files
# =============================================================================
#### As we have already 2010 - 2018 data, add the new data to existing data
precp_product="AORC"
out_dir_lake="C:\\Reseach_CIGLR\\Precipitation\\Precip_products\\AORC\\Precip_Lake"
old_dir_lake=os.path.join(out_dir_lake,'preced')
path_new='C:\\Reseach_CIGLR\\AORC\\2012_regridded'

ij_srl={'west_east':660, 'south_north':622}  # for the index, i = 'west_east', j='south_north'
ij_wsl={'west_east':732, 'south_north':626}  
gauge=['huron_srl','mich_wsl']
index_lake=pd.DataFrame({'Stn_id':gauge, 'west_east':[ij_srl['west_east'],ij_wsl['west_east']], 'south_north': [ij_srl['south_north'], ij_wsl['south_north']]})
old_precip=dict([(key, pd.DataFrame(columns=['Date','Precip'])) for key in gauge]) 

#%%
preced_year=2010
new_year=2012

for igauge in gauge:
    old_precip_lake=pd.read_csv(os.path.join(old_dir_lake,'AORC_Day_'+igauge+'_preced_'+str(preced_year)+'.csv'),delimiter=';',names=['Date','Precip'],skiprows=1).drop_duplicates()
    old_precip[igauge]['Date']=pd.to_datetime(old_precip_lake['Date'])
    old_precip[igauge]['Precip']=old_precip_lake['Precip'].astype(float)
#%
#filename='201204301300.PRECIP_FORCING.nc'  # test

#%%
dict_new_hour=dict([(key, []) for key in gauge]) 
dict_new_precip=dict([(key, []) for key in gauge]) 

#New_hour=dict([(key, pd.DataFrame(columns=['Time','Precip'])) for key in gauge]) 

 #%%
start=datetime.datetime.now()
 
for filename in os.listdir(path_new):
    #%
#    ds=xr.open_dataset(os.path.join(path_new, filename))
    ds=nc4.Dataset(os.path.join(path_new, filename))
    
    if 'Times' and 'valid_time' and 'precip_rate' in ds.variables.keys():
        if ds.variables['precip_rate'].size>0 and ds.variables['valid_time'].size>0:
            for igauge in gauge:
                #%
                i=index_lake['west_east'].loc[index_lake['Stn_id']==igauge]
                j=index_lake['south_north'].loc[index_lake['Stn_id']==igauge]
    #            precip_rate=ds['precip_rate'].isel(Time=0, south_north=int(j.values), west_east=int(i.values)).data # mm s^-1
                ma_precip=ds.variables['precip_rate'][:]  # it is a masked array
                precip_rate=ma_precip.data[0, int(j.values), int(i.values)]
                if precip_rate==ma_precip.fill_value or precip_rate<0:
                    precip_rate=np.nan 
                         
                ma_time=ds.variables['valid_time'][:]
                rain_time=ma_time.data[0]
                if rain_time==ma_time.fill_value or rain_time<0:
                    rain_time=np.nan                   
                
#                New_hour[igauge]=New_hour[igauge].append({'Time': float(rain_time),'Precip': float(precip_rate)},ignore_index=True)
                dict_new_hour[igauge].append(rain_time)
                dict_new_precip[igauge].append(3600*precip_rate)


print (datetime.datetime.now()-start)

#rain_time=datetime.datetime.utcfromtimestamp(rain_time)
#sorted(dict_new_hour[igauge])[:3]
#%%
path_out_new=out_dir_lake+'\\new_version\\'+str(new_year)
date_format='%Y%m%d %H'      # date format for the overlake rain gauge data
freq_in='H'        # gauge data
freq_out='D'           # desired daily data
min_count=24 
begin_date=str(new_year)+'0430 13'
end_date=str(new_year)+'1231 23'
#%%
df_na=pd.DataFrame(columns=['Init_1H_NA','Fra_Init_NA','Resamp_Day_NA','Fra_Resamp_NA', 'begin_time', 'end_time', 'replace_date'], index=gauge)
Total_update=dict([(key, old_precip[key].copy()) for key in gauge])   # for dataframes, you should use copy(), else the original will be modified
 
#%%
for igauge in gauge:
    #%
    df_new_hour = pd.DataFrame(list(zip(dict_new_hour[igauge], dict_new_precip[igauge])), 
               columns =['Time', 'Precip']) 
    df_new_hour['Time']=pd.to_datetime(df_new_hour['Time'], unit='s')
    df_new_hour=df_new_hour.sort_values(by=['Time'])

    hourly_file=os.path.join(path_out_new, str(igauge)+'_'+precp_product+'_hourly_'+str(new_year)+'.csv')
    df_new_hour.to_csv(hourly_file, date_format=date_format, index=False, sep=';')        
    #%
    df_precip=df_new_hour.set_index('Time')
    precip,na_init,na_resamp=resamp_df(df_precip, freq_in, freq_out, min_count, begin_date, end_date, date_format)
    precip.to_csv(os.path.join(path_out_new, str(igauge)+'_'+precp_product+'_Day_'+str(new_year)+'.csv'),date_format=date_format, index=True, sep=';')
    #% Now, replace the old daily precipitation data
    rep_precip=precip.dropna()
    rep_precip.index=rep_precip.index.astype('datetime64[ns]') 
    Total_update[igauge]=Total_update[igauge].set_index('Date')
    Total_update[igauge].update(rep_precip)
    Total_update[igauge].to_csv(os.path.join(path_out_new, str(igauge)+'_'+precp_product+'_Day_updated_'+str(new_year)+'.csv'),date_format=date_format, index=True, sep=';')
    
    #% dnalyse files
    df_na.loc[igauge,'Init_1H_NA']=na_init
    df_na.loc[igauge,'Fra_Init_NA']=na_init/(24*365*9) # 1 Hour time step data
    df_na.loc[igauge,'Resamp_Day_NA']=na_resamp
    df_na.loc[igauge,'Fra_Resamp_NA']=na_resamp/(365*9)
    df_na.loc[igauge,'begin_time']=begin_date
    df_na.loc[igauge,'end_time']=end_date
    df_na.loc[igauge,'replace_date']=len(rep_precip)

df_na.to_csv(os.path.join(path_out_new, precp_product+'_Info_update_'+str(new_year)+'.csv'),sep=';')

#%% 
# =============================================================================
# For overlake rain gages, write files and compare
# =============================================================================
# ==============================================================
#         #% plot obs vs product
#         # =============================================================================
#         begin_date=str(iyear)+'0101 00'
#         end_date=str(iyear)+'1231 23'
#         gauge_data=Read_rain_file(dir_station_obs, gauge_file, str(iyear)+'/01/01 00:00', str(iyear)+'/12/31 23:30', date_format_stn)        
#         #% plot
#         if len(gauge_data)>0:
#             x_obs=[datetime.datetime.strptime(gauge_data[i].split(';')[0], date_format_stn) for i in range(0,len(gauge_data))]
#             y_obs=[2*float(gauge_data[i].split(';')[1]) for i in range(0,len(gauge_data))] # record each 30 minutes, to mm/hr
#         else:
#             x_obs=[]
#             y_obs=[]
        
#         x_AORC = [x[0] for x in AORC_rain_data]
#         y_AORC = [x[1]*3600 for x in AORC_rain_data]  # mm/s to mm/hr
#         #%
#         fig=plt.figure()
#         plt.plot(x_obs,y_obs,'or',x_AORC,y_AORC,'b-')
#         plt.title(str(igage)+'-Hourly-Precipitation-'+str(iyear))
#         plt.xlabel('Date')
#         plt.ylabel('Precipitation (mm/hr)')
#         plt.legend(['station','AORC'])
#         plt.ylim(0,20)
#         plt.xlim(datetime.datetime.strptime(begin_date, date_format),datetime.datetime.strptime(end_date, date_format))
#         plt.show()
#         fig.savefig(out_dir+'\\'+str(igage)+'_hourly_precip.jpg')
        
#         fig2=plt.figure()
#         plt.plot(x_obs,y_obs,'or',x_AORC,y_AORC,'b-')
#         plt.title(str(igage)+'-Hourly-Precipitation-'+str(iyear))
#         plt.xlabel('Date')
#         plt.ylabel('Precipitation (mm/hr)')
#         plt.legend(['station','AORC'])
# #        plt.ylim(0,20)
#         plt.xlim(datetime.datetime.strptime(begin_date,date_format),datetime.datetime.strptime(end_date, date_format))
#         plt.show()
#         fig2.savefig(out_dir+'\\'+str(igage)+'_hourly_precip_noLim.jpg')

#%%
# =============================================================================
# Generate an overall AORC overlake precip data file
# =============================================================================
root_path="C:\\Reseach_CIGLR\\Precipitation\\Precip_products"
precp_product="AORC"
years=list(range(2010,2019))
product_path=os.path.join(root_path,precp_product+'\\Overlake2')
if not os.path.isdir(product_path):
    raise Exception('AORC directory not exist')

gauge=['huron-srl','mich-wsl']
#%
for igage in gauge:
    file_name=str(igage)+'_'+precp_product+'_hourly_'+str(years[0])+'_'+str(years[-1])+'.dat'
    final_file_name=os.path.join(product_path,file_name)
    if os.path.isfile(final_file_name): os.remove(final_file_name)
  
    with open(final_file_name,'w') as final_file:          
        for iyear in years:
            dir_path=os.path.join(product_path,str(iyear))      
            AORC_gage_name=str(igage)+'_'+precp_product+'_hourly.dat'
            AORC_gage_file=os.path.join(dir_path, AORC_gage_name)
            if not os.path.isfile(AORC_gage_file):
                raise Exception('AORC file not exist')
                break
            for line in open(AORC_gage_file,'r'):
                final_file.write(line)
#%%
# =============================================================================
# Compare the AORC overlake with the gauge overlake, year by year       
# =============================================================================
#### Generate daily precip data from the hourly data
##%%
dir_path='C:\\Reseach_CIGLR\\Precipitation\\Precip_products\\AORC\\Overlake2'
precp_product="AORC"

years=list(range(2010,2019))
gauge=['huron-srl','mich-wsl']
date_format='%Y%m%d %H'      # date format for the overlake rain gauge data
freq_in='H'        # gauge data
freq_out='D'           # desired daily data
min_count=24   # set the value to nan for any 'freq_out' which has fewer than 'mean_count' data in 'freq_in'   
#%% NaN yearly analysis results dataframe
index_anal=pd.period_range(str(years[0]),str(years[-1]),freq='Y')
# creat dictionaries of dataframe
df_anal={}
df_precip={}
for igage in gauge:
    rain_file=os.path.join(dir_path,str(igage)+'_'+precp_product+'_hourly_'+str(years[0])+'_'+str(years[-1])+'.dat')
    if not os.path.isfile(rain_file): raise Exception('Rain file not exist')
    
    df_anal[igage]=pd.DataFrame(index=index_anal,columns=['Init_1H_NA','Fra_Init_NA','Resamp_Day_NA','Fra_Resamp_NA'])
    df_precip[igage]=pd.DataFrame(columns=['Precip'])
    df_anal[igage].index.name='Years'
    df_precip[igage].index.name='Date'
    
    for iyear in years:
        begin_date=str(iyear)+'0101 0'
        end_date=str(iyear)+'1231 23'        # should correspond to the dateformat
        precip,na_init,na_resamp=resamp_precip(freq_in, freq_out, rain_file, min_count, begin_date, end_date, date_format)
        df_anal[igage].loc[str(iyear),'Init_1H_NA']=na_init
        df_anal[igage].loc[str(iyear),'Fra_Init_NA']=na_init/(24*365) # 1 Hour time step data
        df_anal[igage].loc[str(iyear),'Resamp_Day_NA']=na_resamp
        df_anal[igage].loc[str(iyear),'Fra_Resamp_NA']=na_resamp/365        
        df_precip[igage]=df_precip[igage].append(precip*3600)          # accumulated mm/s to total mm rain in 1 day
    
    df_anal[igage].to_csv(os.path.join(dir_path,'Anal_'+str(igage)+'_'+precp_product+'_'+str(years[0])+'_'+str(years[-1])+'.csv'),sep=';')
    df_precip[igage].to_csv(os.path.join(dir_path,'Precip_Day_'+str(igage)+'_'+precp_product+'_'+str(years[0])+'_'+str(years[-1])+'.csv'),sep=';')   

#%% plot pictures of daily results and calculate the MAE and RMSE
dir_produc='C:\\Reseach_CIGLR\\Precipitation\\Precip_products\\AORC\\Overlake2'
dir_obs='C:\\Reseach_CIGLR\\Precipitation\\station_observations\\ReCON_stations'
precp_product="AORC"

years_prod=list(range(2010,2019))
years_obs=list(range(2013,2019))
gauge=['huron-srl','mich-wsl']
#%% MAE, RMSE and annual diff analysis
index_anal=[str(iyear) for iyear in years]
index_anal.append('Total')

dict_anal=dict([(key, pd.DataFrame(index=index_anal, columns=gauge)) for key in ['MAE','RMSE','Anual_diff']]) 
#%
for igage in gauge:
    rain_obs=pd.read_csv(os.path.join(dir_obs,'Precip_Day_'+str(igage)+'_'+str(years_obs[0])+'_'+str(years_obs[-1])+'.csv'),delimiter=';',names=['Date','Precip'],skiprows=1)
    rain_prod=pd.read_csv(os.path.join(dir_produc,'Precip_Day_'+str(igage)+'_'+precp_product+'_'+str(years_prod[0])+'_'+str(years_prod[-1])+'.csv'),delimiter=';',names=['Date','Precip'],skiprows=1)

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
        
        if float(obs_compare.sum())>0:
            anual_diff=(prod_compare.sum()-obs_compare.sum())/obs_compare.sum()
        else: anual_diff=np.nan
        
        #% MAE, RMSE
        MAE, RMSE=MAE_RMSE_Diff(obs_compare,prod_compare)
        dict_anal['MAE'].loc[str(iyear),igage]=float(MAE)
        dict_anal['RMSE'].loc[str(iyear),igage]=float(RMSE)
        dict_anal['Anual_diff'].loc[str(iyear),igage]=float(anual_diff)
        #% plot
        ax=prod_compare.plot(linestyle='-',color='b',title=str(igage)+'-Daily-Precipitation-'+str(iyear))
        obs_compare.plot(ax=ax,linestyle=':',color='r',lw=4)
        ax.set(xlabel='Date',ylabel='Precipitation (mm/day)')
        ax.legend(labels=[precp_product,'Observation'],loc='upper right')
        textstr = 'MAE=%.3f\nRMSE=%.3f\n(AORC-Obs)/Obs=%.3f\n'%(MAE,RMSE,anual_diff)
        ax.annotate(textstr, xy=(0.01, 0.8), xycoords="axes fraction")
        #%
        #ax.figure.autofmt_xdate()
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.tick_params(axis='x', rotation=70)
        #plt.show()
        fig = ax.get_figure()
        fig.savefig(os.path.join(out_dir,str(igage)+'_daily_'+precp_product+'.jpg'))
#%
    total_begin_date=str(years_obs[0])+'0101'
    total_end_date=str(years_obs[-1])+'1231' 
    total_obs_compare=rain_obs.loc[(pd.to_datetime(rain_obs['Date'])>=pd.to_datetime(total_begin_date)) & (pd.to_datetime(rain_obs['Date'])<=pd.to_datetime(total_end_date))].drop_duplicates()
    total_prod_compare=rain_prod.loc[(pd.to_datetime(rain_prod['Date'])>=pd.to_datetime(total_begin_date)) & (pd.to_datetime(rain_prod['Date'])<=pd.to_datetime(total_end_date))].drop_duplicates()
    #% prod_compare.dtypes
    total_obs_compare=total_obs_compare['Precip']
    total_prod_compare=total_prod_compare['Precip']
    #% MAE, RMSE, diff
    total_MAE, total_RMSE=MAE_RMSE_Diff(total_obs_compare,total_prod_compare)
    if float(total_obs_compare.sum())>0:       
        total_anual_diff=(total_prod_compare.sum()-total_obs_compare.sum())/total_obs_compare.sum()
    else: total_anual_diff=np.nan
    #%
    dict_anal['MAE'].loc['Total',igage]=float(total_MAE)        
    dict_anal['RMSE'].loc['Total',igage]=float(total_RMSE)         
    dict_anal['Anual_diff'].loc['Total',igage]=float(total_anual_diff)    # precipitation annual difference 
 
#%
dict_anal['MAE'].to_csv(os.path.join(dir_produc,'MAE_Lake_AORC_'+str(years[0])+'_'+str(years[-1])+'.csv'),sep=';')
dict_anal['RMSE'].to_csv(os.path.join(dir_produc,'RMSE_Lake_AORC_'+str(years[0])+'_'+str(years[-1])+'.csv'),sep=';')
dict_anal['Anual_diff'].to_csv(os.path.join(dir_produc,'Anual_diff_Lake_AORC_'+str(years[0])+'_'+str(years[-1])+'.csv'),sep=';')         
    

#%%
# =============================================================================
# For overland stations, hourly data have been written in csv files, resample these data to daily data
# =============================================================================
years=list(range(2010,2019))
hourly_path="C:\\Reseach_CIGLR\\Precipitation\\Precip_products\\AORC\\Land\\Hourly"
daily_path="C:\\Reseach_CIGLR\\Precipitation\\Precip_products\\AORC\\Land\\Daily"
stn_info_file='C:\\Reseach_CIGLR\\Precipitation\\station_observations\\stations_since_2010.txt'
col_names=['Stn_id','Stn_name','Lon','Lat']
stn_info= pd.read_csv(stn_info_file, sep=r'\s\s\s+', header=None, skipinitialspace=True,engine='python', names=col_names)
#%%
date_format='%Y%m%d %H'      # date format for the overlake rain gauge data
freq_in='H'        # gauge data
freq_out='D'           # desired daily data
min_count=24   # set the value to nan for any 'freq_out' which has fewer than 'mean_count' data in 'freq_in'  

begin_date=str(years[0])+'0101 0'
end_date=str(years[-1])+'1231 23'   

#%%
df_na=pd.DataFrame(columns=['Init_1H_NA','Fra_Init_NA','Resamp_Day_NA','Fra_Resamp_NA'], index=stn_info['Stn_id'].astype(str).tolist())
    
for istn in range(len(stn_info)):
    #%
    stn_name=str(stn_info['Stn_id'].iloc[istn])
    filename='Precip_Hour_'+str(stn_name)+'_AORC_'+str(years[0])+'_'+str(years[-1])+'.csv'
    rain_hourly=pd.read_csv(os.path.join(hourly_path,filename),delimiter=';',names=['Date','Precip'],skiprows=1)
    df_precip=rain_hourly.sort_values(by=['Date']).set_index('Date')
    df_precip.index=pd.to_datetime(df_precip.index)
    precip,na_init,na_resamp=resamp_df(df_precip, freq_in, freq_out, min_count, begin_date, end_date, date_format)
    
    df_na.loc[stn_name,'Init_1H_NA']=na_init
    df_na.loc[stn_name,'Fra_Init_NA']=na_init/(24*365*9) # 1 Hour time step data
    df_na.loc[stn_name,'Resamp_Day_NA']=na_resamp
    df_na.loc[stn_name,'Fra_Resamp_NA']=na_resamp/(365*9)
    precip=3600*precip    
    
    precip.to_csv(os.path.join(daily_path,'AORC_Day_'+stn_name+'_'+str(years[0])+'_'+str(years[-1])+'.csv'),sep=';')
#% the max value of the index column 0 and 1, 
df_na.to_csv(os.path.join("C:\\Reseach_CIGLR\\Precipitation\\Precip_products\\AORC\\Land",'NA_OL_AORC_'+str(years[0])+'_'+str(years[-1])+'.csv'),sep=';')

#%%
    