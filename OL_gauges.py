#=================================
# This Python can read the rain gauge data of the 1287 overland stations which has data since 2010:
# The data is recorded in the .csv files, 
# Station infos are registred in stations_since_2010.txt, i,j for AORC info in the aorc_stninfo_precip_ij.txt

# This script read only precipitation data
# Edited by Yi Hong, Nov. 15 2019
# Modified by Yi Hong, Feb. 24 2020, for adding new 2019 data
# Modified by Yi Hong, Apr. 21 2020, add analysis of canadian side data
#=================================

#%% import packages
import numpy as np
import pandas as pd
import xarray as xr
import datetime
import os
import matplotlib.pyplot as plt
from precip_functions import Read_rain_file, resamp_df,map_area,stn_filter

os.environ["PROJ_LIB"] = "C:\\ProgramData\\Anaconda3\\Library\\share"; #path of the proj for basemap
from mpl_toolkits.basemap import Basemap
#%%
# =============================================================================
# To be done in the first time, read rain gage data and write to a separated file
# =============================================================================
# =============================================================================
dir_stn='C:\\Reseach_CIGLR\\Precipitation\\station_observations'
dir_data=dir_stn+'\\tim_rain_data'
dir_anal=dir_stn+'\\anal_2010_2019'

dir_OL_stn=dir_stn+'\\OL_stn2'
if not os.path.isdir(dir_OL_stn):  
    os.mkdir(dir_OL_stn)

#%
info_file=os.path.join(dir_stn,'stations_since_2010.txt')
#lonlat_file=os.path.join(dir_stn,'stninfo_precip_lonlat.txt')
#%
col_names=['Stn_id','Stn_name','Lon','Lat']
stn_info= pd.read_csv(info_file, sep=r'\s\s\s+', header=None, skipinitialspace=True,engine='python', names=col_names)

#%%
date_format='%Y-%m-%d'
#begin_date='2010-01-01'
#end_date='2018-12-31'
years=list(range(2010,2020)) 
freq_in='D'        # gauge data
freq_out='D'           # desired daily data
min_count=1 
#%% Record one analyse file for all the OL stations per year and total, and separate precip files
index_anal=pd.period_range(str(years[0]),str(years[-1]),freq='Y')
col_names=list(stn_info.loc[:,'Stn_id'])

df_anal=pd.DataFrame(index=index_anal,columns=col_names)     # the annual daily NA for each station
df_precip={}
#%%
for istn in range(len(stn_info)):
    #%
    file_name='met_'+str(stn_info['Stn_id'].iloc[istn])+'.csv'  
    if not os.path.join(dir_data,file_name):
        raise Exception('Station data file is not in the folder')
        break
    #%
    data_col_names=['Date','AirTempMax','AirTempMin','Precip','AirTemp','Dewpoint','Windspeed','CloudCover']
    stn_data=pd.read_csv(os.path.join(dir_data,file_name), sep=',', skiprows=9, header=None, engine='python', names=data_col_names)        
    #%
    precip_data=stn_data.loc[:,['Date','Precip']].drop_duplicates()
    precip_data['Date']=pd.to_datetime(precip_data['Date'])
    precip_data['Precip']=pd.to_numeric(precip_data['Precip'],errors='coerce')
    precip_data.loc[precip_data['Precip']<0,'Precip']=np.nan 
    precip_data=precip_data.set_index('Date').sort_index()    
    #%
    df_precip[str(stn_info['Stn_id'].iloc[istn])]=pd.DataFrame(columns=['Precip'])
    #%  
    for iyear in years:
        #%
        begin_date=str(iyear)+'-01-01'
        end_date=str(iyear)+'-12-31'   # data should correspond to the date format
        precip_query, na_in, na_out=resamp_df(precip_data, freq_in, freq_out, min_count, begin_date, end_date, date_format)
    
        df_anal.loc[str(iyear),stn_info.loc[istn,'Stn_id']]=na_out/365   # percentage of the annual NA data
        df_precip[str(stn_info['Stn_id'].iloc[istn])]=df_precip[str(stn_info['Stn_id'].iloc[istn])].append(precip_query)
           
        ## precip data for each stations
        df_precip[str(stn_info['Stn_id'].iloc[istn])].to_csv(os.path.join(dir_OL_stn,'Precip_Day_'+str(stn_info['Stn_id'].iloc[istn])+'_'+str(years[0])+'_'+str(years[-1])+'.csv'),sep=';')        
#%%
df_anal.to_csv(os.path.join(dir_anal,'NA_OL_Stns_'+str(years[0])+'_'+str(years[-1])+'.csv'),sep=';')
#% Total NA
anal_file=os.path.join(dir_anal,'NA_OL_Stns_'+str(years[0])+'_'+str(years[-1])+'.csv')
df_anal=pd.read_csv(anal_file, sep=';',index_col=0, header=0)
#%
df_anal2=pd.DataFrame(index=['Total_NA2'],columns=col_names) 
#%%        
for istn in range(len(stn_info)):
    file_name='met_'+str(stn_info['Stn_id'].iloc[istn])+'.csv'  

    data_col_names=['Date','AirTempMax','AirTempMin','Precip','AirTemp','Dewpoint','Windspeed','CloudCover']
    stn_data=pd.read_csv(os.path.join(dir_data,file_name), sep=',', skiprows=9, header=None, engine='python', names=data_col_names)        
    #%
    precip_data=stn_data.loc[:,['Date','Precip']].drop_duplicates()
    precip_data['Date']=pd.to_datetime(precip_data['Date'])
    precip_data=precip_data.set_index('Date').sort_index()    
    #%
    df_precip[str(stn_info['Stn_id'].iloc[istn])]=pd.DataFrame(columns=['Precip'])        
    begin_date=str(years[0])+'-01-01'
    end_date=str(years[-1])+'-12-31'   # data should correspond to the date format
    precip_query, na_in, na_out=resamp_df(precip_data, freq_in, freq_out, min_count, begin_date, end_date, date_format)
    
    df_anal2.loc['Total_NA2',str(stn_info['Stn_id'].iloc[istn])]=na_out/(365*len(years))
#%%       
df_anal=df_anal.append(df_anal2)
df_anal.to_csv(os.path.join(dir_anal,'NA_OL_Stns_'+str(years[0])+'_'+str(years[-1])+'.csv'),sep=';')
#%% plot observation NaN fractions
#fig, ax = plt.subplots()
#df_anal.iloc[-1].value_counts().plot(ax=ax, kind='bar')
for iindex in df_anal.index:
    #%
#    iindex='Total_NA2'
    ax=df_anal.loc[iindex].plot(kind='hist',title=str(iindex)+'_NaN_fraction')
    ax.set(xlabel='NaN fraction',ylabel='Number of stations')
    fig = ax.get_figure()
    fig.savefig(os.path.join(dir_anal,'Figures\\'+str(iindex)+'_NaN_fraction.jpg'))
    plt.close(fig)
#%%   
# =============================================================================
# Read data from csv files, and plot  
# =============================================================================
anal_file=os.path.join(dir_anal,'NA_OL_Stns_'+str(years[0])+'_'+str(years[-1])+'.csv')
df_anal=pd.read_csv(anal_file, sep=';',index_col=0, header=0)

shorline_file=os.path.join(dir_stn,'greatlakes.shoreline.dat2')
min_Lon,max_Lon,min_Lat,max_Lat=map_area(shorline_file)

#%% plot maps
for iindex in df_anal.index:
#%%
iindex='Total_NA2'
limit_na=0.15  
    #%%
    fig, ax = plt.subplots()
    map = Basemap(projection='merc',resolution='i',lat_0=45,lon_0=-83, area_thresh=1000.0, llcrnrlon=min_Lon-2,llcrnrlat=min_Lat-2,urcrnrlon=max_Lon+2,urcrnrlat=max_Lat+2) #lat_0=45,lon_0=-83,
    parallels = np.arange(0.,81,5.)
    map.drawparallels(parallels,labels=[True,False,False,False],linewidth=0.1)  ## fontsize=ftz,
    meridians = np.arange(10.,351.,5.)
    map.drawmeridians(meridians,labels=[False,False,False,True],dashes=[8,1],linewidth=0.1)  ##fontsize=ftz,
    map.drawcoastlines()    
    
    #% add points in the figure
    stn_out, df_anal_out=stn_filter(iindex,stn_info,limit_na,df_anal)
    #% Record the new station file
    stn_out.to_csv(os.path.join(dir_anal,'stations_NaN_Fraction_'+str(limit_na)+'.csv'),sep=';',index=False)
    #%
#    c=df_anal.loc[iindex].apply(lambda x: x if x <= limit_na else np.nan)
    #%
    x,y=map(list(stn_out['Lon']),list(stn_out['Lat'])) 
    #scatter = map.scatter(x,y,c=df_anal.loc[df_anal.index[0]],s=10,vmin=0, vmax=1, cmap='RdBu')
    scatter = map.scatter(x,y,s=10,c=df_anal_out.loc[iindex],vmin=0, vmax=limit_na, cmap='summer')
    cbar=plt.colorbar()
    cbar.set_label('NaN fraction',labelpad=10, y=0.45, rotation=270)
    plt.title("NaN fraction of rain gauges in "+str(iindex)+'_'+str(limit_na))
    plt.annotate(str(len(stn_out))+' Stations', xy=(0.7, 0.9), xycoords="axes fraction")
    plt.savefig(os.path.join(dir_anal,'Figures\\'+str(iindex)+'_Map_NaN_Fraction_'+str(limit_na)+'.jpg'))
    plt.show() 

#%
#for iindex in df_anal.index:
#    scatter.set_array(df_anal.loc[iindex])
#    ttl.set_text("NaN fraction of rain gauges in "+str(iindex))
#    plt.savefig(os.path.join(dir_stn,'Figures\\'+str(iindex)+'_Map_NaN_Fraction.jpg'))
#    plt.show() 
#%%
# =============================================================================
# Check the ajusted canadian data points
# =============================================================================
# =============================================================================
dir_stn='C:\\Reseach_CIGLR\\Precipitation\\station_observations'
info_file=os.path.join(dir_stn,'stations_since_2010.txt')
col_names=['Stn_id','Stn_name','Lon','Lat']
stn_info= pd.read_csv(info_file, sep=r'\s\s\s+', header=None, skipinitialspace=True,engine='python', names=col_names)
shorline_file=os.path.join(dir_stn,'greatlakes.shoreline.dat2')
min_Lon,max_Lon,min_Lat,max_Lat=map_area(shorline_file)
#%%
ca_info_file=os.path.join(dir_stn,'Ajusted_canadian\\StationSIS_3346.xls')
stn_ca_info=pd.read_excel(ca_info_file)
#%%
end_count=stn_ca_info['End year'].value_counts()
end_count.to_csv(os.path.join(dir_stn,'Ajusted_canadian\\End_year_count.csv'),sep=';',index=True) 
#%% within great lake region
flag1=stn_ca_info['Latitude (Decimal Degrees)'] >= min_Lat
flag2=stn_ca_info['Latitude (Decimal Degrees)'] <= max_Lat
flag3=stn_ca_info['Longitude (Decimal Degrees)'] >= min_Lon
flag4=stn_ca_info['Longitude (Decimal Degrees)'] <= max_Lon
#%%
stn_glr=stn_ca_info.loc[flag1 & flag2 & flag3 & flag4]
stn_glr=stn_glr.sort_values(by=['End year'], ascending=False)
stn_glr.to_csv(os.path.join(dir_stn,'Ajusted_canadian\\Stn_GLR.csv'),sep=';',index=False) 

stn_glr_2010=stn_glr.loc[stn_glr['End year']>200912]
#stn_glr_2010.to_csv(os.path.join(dir_stn,'Ajusted_canadian\\Stn_GLR_2010.csv'),sep=';',index=False) 

#%%
dir_ajust=os.path.join(dir_stn,'Ajusted_canadian\\stns3346_dataQCd_fixedTflags_QCd202003')
#%%
stn_2010_new=stn_glr_2010.copy().reset_index()
stn_2010_new['NA_Fr']=''
#%%
for i in range(0,len(stn_glr_2010)):
    #%
    filename=os.path.join(dir_ajust, stn_glr_2010['Climate ID'].iloc[i]+'_QCdRSP.txt')
    
    f=open(filename, 'r', errors=None)
    txt=f.readlines()
    f.close()
    #%
    df_ca=pd.DataFrame(columns=['Date','Precip'])
    for line in txt[1:len(txt)]:
        #%
        year=line[8:12].split()[0]
        mon=line[12:14].split()[0]
        day=line[14:16].split()[0]

        time=datetime.datetime.strptime(year+mon+day,'%Y%m%d')  
        precip=float(line[103:111].split()[0])
        df_ca=df_ca.append({'Date':time,'Precip':precip},ignore_index=True)
  #%
    df_ca['Date']=pd.to_datetime(df_ca['Date'])
    df_ca_2010=df_ca.loc[df_ca['Date']>=datetime.datetime.strptime('20100101','%Y%m%d')]
    #%
    df_ca_2010['Precip'][df_ca_2010['Precip']<0]=np.nan    
    df_ca_2010.to_csv(os.path.join(dir_stn,'Ajusted_canadian\\Stn_GLR\\R_'+stn_glr_2010['Climate ID'].iloc[i]+'.csv'), date_format='%Y%m%d', sep=';',index=False)
    Fra_na=df_ca_2010['Precip'].isna().sum()/len(df_ca_2010)
    stn_2010_new.loc[i,'NA_Fr']=Fra_na
#%
stn_2010_new.to_csv(os.path.join(dir_stn,'Ajusted_canadian\\Stn_GLR_2010_NA.csv'),sep=';',index=False) 
#%%
stn_2010_new=pd.read_csv(os.path.join(dir_stn,'Ajusted_canadian\\Stn_GLR_2010_NA.csv'), sep=';')
#%%
ax=stn_2010_new['NA_Fr'].loc[stn_2010_new['End year']>201601].plot(kind='hist',title='Missing_data_fraction_Great_Lake_Region')
ax.set(xlabel='Missing_data_fraction',ylabel='Number of stations')
fig = ax.get_figure()
fig.savefig(os.path.join(dir_stn,'Ajusted_canadian\\Missing_data_fraction.jpg'))
plt.close(fig)


#%%
df_ajust=pd.DataFrame(columns=['Clim_id','Begin_year','End_year'])

for name in os.listdir(dir_ajust):
#    name='101HFNE_QCdRSP.txt'
    filename=os.path.join(dir_ajust, name)
    #%
    f=open(filename, 'r', errors=None)
    txt=f.readlines()
    l_begin=txt[1]
    l_end=txt[len(txt)-1]
    f.close()
    #
    sp_begin=l_begin.split()
    sp_end=l_end.split()
    df_ajust=df_ajust.append({'Clim_id':sp_begin[0], 'Begin_year': sp_begin[1][:4], 'End_year':sp_end[1][:4]}, ignore_index=True)
#%%
    df_ajust['Begin_year']=df_ajust['Begin_year'].astype(int)
    df_ajust['End_year']=df_ajust['End_year'].astype(int)
    df_ajust=df_ajust.sort_values(by=['End_year'],ascending=False)
#%
    df_ajust.to_csv(os.path.join(dir_stn,'Ajusted_canadian\\Begin_end.csv'),sep=';',index=False)
#%%
stn_ca_2016=stn_ca_info.loc[stn_ca_info['End year']>201601]   
    
#%%    
fig, ax = plt.subplots()
map = Basemap(projection='merc',resolution='i',lat_0=45,lon_0=-83, area_thresh=1000.0, llcrnrlon=min_Lon-2,llcrnrlat=min_Lat-2,urcrnrlon=max_Lon+2,urcrnrlat=max_Lat+2) #lat_0=45,lon_0=-83,
parallels = np.arange(0.,81,5.)
map.drawparallels(parallels,labels=[True,False,False,False],linewidth=0.1)  ## fontsize=ftz,
meridians = np.arange(10.,351.,5.)
map.drawmeridians(meridians,labels=[False,False,False,True],dashes=[8,1],linewidth=0.1)  ##fontsize=ftz,
map.drawcoastlines()    

#% add points in the figure
x2,y2=map(list(stn_info['Lon']),list(stn_info['Lat'])) 
scatter = map.scatter(x2,y2,s=5,c='black',marker='*')
x,y=map(list(stn_2010_new['Longitude (Decimal Degrees)'].loc[stn_2010_new['End year']>201601]),list(stn_2010_new['Latitude (Decimal Degrees)'].loc[stn_2010_new['End year']>201601])) 
scatter = map.scatter(x,y,s=10,c=stn_2010_new['NA_Fr'].loc[stn_2010_new['End year']>201601],vmin=0, vmax=0.3, cmap='summer',marker='^')
cbar=plt.colorbar()
cbar.set_label('NaN fraction',labelpad=10, y=0.45, rotation=270)

plt.title("Ajusted Canadian Stations")
plt.savefig(os.path.join(dir_stn,'Ajusted_canadian\\Ajusted_CA_stations_NA.jpg'))
plt.show() 

