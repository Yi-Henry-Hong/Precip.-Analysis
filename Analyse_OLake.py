# This scripts is to analyze the recorded data of the overlake stations
# Edited by Yi Hong, Nov. 25 2019
# Modified by Yi Hong, Mars 10 2020, add the analysis for different rainfall products

#import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
#from pandas.plotting import register_matplotlib_converters
import os
import numpy as np
import datetime
import calendar
#import re
#import matplotlib.dates as mdates
from precip_functions import MAE_RMSE_Diff, Relat_Diff

#os.environ["PROJ_LIB"] = "C:\\ProgramData\\Anaconda3\\Library\\share"; #path of the proj for basemap
#from mpl_toolkits.basemap import Basemap


#%%
# =============================================================================
# Data directories      
# =============================================================================
#product=['AORC','CaPA','MPE','Mrg','HRRR']
product=['AORC','CaPA','MPE','Mrg']
dir_obs='C:\\Reseach_CIGLR\\Precipitation\\station_observations\\ReCON_stations'
dir_out='C:\\Reseach_CIGLR\\Precipitation\\Precip_products\\Analysis'


#%% Initialize the dictionary of multi-index dataframes
gauge=['huron_srl','mich_wsl']
list_var=['MAE','RMSE','Diff','Relat_diff']
years_obs=list(range(2013,2020))
years_compare=list(range(2013,2020))

month_names=list(map(lambda x: datetime.datetime.strptime(str(x), '%m').strftime('%b'), list(range(1,13))))

mul_col=pd.MultiIndex.from_product([month_names, list_var], names=['Month','Var'])
mul_index=pd.MultiIndex.from_product([years_compare, gauge], names=['Year','Gauge'])
anal_precip=dict([(prod, pd.DataFrame(columns=mul_col, index=mul_index)) for prod in product]) 

#%% Monthly analysis month by month
for igage in gauge:
#%
    rain_obs=pd.read_csv(os.path.join(dir_obs,'Precip_Day_'+str(igage)+'_2013_2019.csv'),delimiter=';',names=['Date','Precip'],skiprows=1)
    
    for iprod in product:
        dir_prod='C:\\Reseach_CIGLR\\Precipitation\\Precip_products\\'+iprod+'\\Precip_Lake'         
        if iprod == 'HRRR': years_prod=list(range(2014,2020))
        else: years_prod=list(range(2010,2020))
        rain_prod=pd.read_csv(os.path.join(dir_prod, iprod+'_Day_'+str(igage)+'_'+str(years_prod[0])+'_'+str(years_prod[-1])+'.csv'),delimiter=';',names=['Date','Precip'],skiprows=1)   
            
        for iyear in years_compare:
            #%
            for imonth in range(1,13):
                #%
                begin_time=datetime.datetime.strptime(str(iyear)+'%02d' %imonth+'01','%Y%m%d')
                end_time=datetime.datetime.strptime(str(iyear)+'%02d' %imonth+ str(calendar.monthrange(iyear, imonth)[1]) ,'%Y%m%d')
                rain_obs['Date']=pd.to_datetime(rain_obs['Date'])
                obs_comp=rain_obs.loc[(rain_obs['Date']>=begin_time) & (rain_obs['Date']<=end_time)].reset_index()['Precip']
                rain_prod['Date']=pd.to_datetime(rain_prod['Date'])
                
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
#%
                MAE, RMSE, Diff=MAE_RMSE_Diff(prod_comp,obs_comp)
                Relat_diff=Relat_Diff(prod_comp,obs_comp)
                #%
                anal_precip[iprod].loc[(iyear, igage),month_names[imonth-1]]=[MAE,RMSE,Diff,Relat_diff]
               
# =============================================================================
#%% Total monthly analysis: Real monthly analysis, accumulated monthly rains
# =============================================================================
start=datetime.datetime.now()

anal_mon=dict([(prod, pd.DataFrame(columns=mul_col, index=gauge)) for prod in product]) 

for igage in gauge:
#%
    rain_obs=pd.read_csv(os.path.join(dir_obs,'Precip_Day_'+str(igage)+'_2013_2019.csv'),delimiter=';',names=['Date','Precip'],skiprows=1)
    
    for iprod in product:
        #%
        dir_prod='C:\\Reseach_CIGLR\\Precipitation\\Precip_products\\'+iprod+'\\Precip_Lake'       
        if iprod == 'HRRR': years_prod=list(range(2014,2020))
        else: years_prod=list(range(2010,2020))
        rain_prod=pd.read_csv(os.path.join(dir_prod, iprod+'_Day_'+str(igage)+'_'+str(years_prod[0])+'_'+str(years_prod[-1])+'.csv'),delimiter=';',names=['Date','Precip'],skiprows=1)   
        rain_prod['Date']=pd.to_datetime(rain_prod['Date']) 
        #%        
        for imonth in range(1,13):
            #%
            obs_comp=pd.Series([])
            prod_comp=dict([(prod, pd.Series([])) for prod in product])
            number_days=calendar.monthrange(2019, imonth)[1]          # ignor leap years
            
            for iyear in years_compare:
                begin_time=datetime.datetime.strptime(str(iyear)+'%02d' %imonth+'01','%Y%m%d')
                end_time=datetime.datetime.strptime(str(iyear)+'%02d' %imonth+ str(calendar.monthrange(iyear, imonth)[1]) ,'%Y%m%d')
                rain_obs['Date']=pd.to_datetime(rain_obs['Date'])
                obs=rain_obs.loc[(rain_obs['Date']>=begin_time) & (rain_obs['Date']<=end_time)].reset_index()['Precip']
                obs_comp=obs_comp.append(obs)
                #%               
                if (rain_prod['Date'].iloc[0]<=begin_time) & (rain_prod['Date'].iloc[-1]>=end_time):
                    prod_comp_mon=rain_prod.loc[(rain_prod['Date']>=begin_time) & (rain_prod['Date']<=end_time)].reset_index()['Precip']
                else:
                    compare_date = pd.date_range(begin_time,end_time,freq='D')
                    df_compare=pd.DataFrame(index=compare_date, columns=['Precip'])                    
                    temp_prod=rain_prod[rain_prod['Date'].isin(compare_date)]
                    #%
                    if len(temp_prod)>0:
                        for i in range(len(temp_prod)):
                            df_compare.loc[temp_prod['Date'].iloc[i], 'Precip']=temp_prod['Precip'].iloc[i]                    
                    prod_comp_mon=df_compare['Precip']
                
                prod_comp[iprod]=prod_comp[iprod].append(prod_comp_mon)            
    
        #% After calculation, calculate the total monthly analysis
            MAE, RMSE, Diff=MAE_RMSE_Diff(prod_comp[iprod],obs_comp)
            mon_count=(prod_comp[iprod].reset_index(drop=True)-obs_comp.reset_index(drop=True)).count()
            Diff=Diff/(mon_count/number_days)       # monthly average difference
            Relat_diff=Relat_Diff(prod_comp[iprod],obs_comp)
            anal_mon[iprod].loc[igage,month_names[imonth-1]]=[MAE,RMSE,Diff,Relat_diff]


print (datetime.datetime.now()-start)     
 
#%% 
for iprod in product:
    anal_precip[iprod].to_csv(os.path.join(dir_out,'OLake\\Mon_Ind_Anal_'+iprod+'.csv'),sep=';')
    anal_mon[iprod].to_csv(os.path.join(dir_out,'OLake\\Mon_Anal_'+iprod+'.csv'),sep=';')

#%% boxplot of each var
ylim={'MAE':[0,5],'RMSE':[0,10],'Diff':[-25,125],'Relat_diff':[-1,15]}
df_anal_mon=dict([(igage, pd.DataFrame(columns=mul_col, index=product)) for igage in gauge])
#%
for iprod in product:
    #%
    for ivar in list_var:  
#% 
        df_anal_ind=anal_precip[iprod].xs(ivar,axis=1,level=1,drop_level=True).astype(float)
        df_anal_mon[gauge[0]].loc[iprod]=anal_mon[iprod].xs(gauge[0],axis=0,drop_level=True).astype(float)
        df_anal_mon[gauge[1]].loc[iprod]=anal_mon[iprod].xs(gauge[1],axis=0,drop_level=True).astype(float)
        #%%
        if ivar in ['MAE', 'RMSE']:
            text_title= 'Boxplots of '+iprod+' Vs Obs. '+ivar        
        elif ivar=='Diff':
            text_title= 'Boxplots of mean monthly difference ('+iprod+'-Obs.)'  
        else:
            text_title= 'Boxplots of related monthly difference (['+iprod+'-Obs.]/Obs.)'                    
            
        fig=plt.figure()
        ax = df_anal_ind.boxplot() 
        ax.set_ylim(ylim[ivar])
        plt.title(text_title)
        #%
        fig.savefig(os.path.join(dir_out,'Figures\\OLake\\Boxplots_'+ivar+'_'+iprod+'.jpg'))
        fig.show()                                  
   
    #%% barplots
RMSE_all=pd.read_csv(os.path.join(dir_out, 'OLake\\RMSE_Prod.csv'), sep=';')
#%%
for igage in gauge:
    textstr=''
    for iprod in product:
        RMSE=RMSE_all[igage].loc[RMSE_all['Product']==iprod]
        textstr = textstr+'RMSE_'+iprod+'=%.2f\n'%RMSE
                 
    for ivar in list_var: 
#    ivar='Relat_diff'
        if ivar in ['MAE', 'RMSE']:
            text_title= 'Barplots of '+ivar +' ' +igage 
            text_y=ivar
        elif ivar=='Diff':
            text_title= 'Barplots of mean monthly difference (mm) '+ igage 
            text_y='Prod.-Obs.'
        else:
            text_title= 'Barplots of related monthly difference ' + igage
            text_y='(Prod.-Obs.)/Obs.'
        #%   
        ax = df_anal_mon[igage].xs(ivar,axis=1,level=1,drop_level=True).T.plot(kind='bar', figsize=(9,6), fontsize=16)
        ax.legend(labels=product,loc='upper center',ncol=2,fontsize=16)
        ax.set_title(text_title,fontsize=16, weight='bold')
        ax.set_xlabel("Month", fontsize=16, weight='bold')
        ax.set_ylabel(text_y, fontsize=16, weight='bold')
        ax.annotate(textstr, xy=(0.01, 0.65), xycoords="axes fraction",fontsize=14)
        #%
        plt.savefig(os.path.join(dir_out,'Figures\\OLake\\Barplots2_'+ivar+'_'+igage+'.jpg'),bbox_inches='tight')
        plt.show()


#%%




















# =============================================================================
# The two overlake stations from the overland files
# =============================================================================
add_gage=pd.read_csv('C:\\Reseach_CIGLR\\Precipitation\\station_observations\\OLake_id.dat', sep=',', header=None, skipinitialspace=True,names=['Stn_id','Stn_name','Lat','Lon','Begin','End'])
#%
info_recon={'Stn_id':['ReCon-srl','ReCon-wsl'],'Stn_name':['huron-srl','mich-wsl'],'Lat':[45.773200,45.842217],'Lon':[-84.136700,-85.135283],'Begin':['2013-06-08','2013-09-25'],'End':['2018-12-31','2018-12-31']} 
all_gage=add_gage.append(pd.DataFrame(info_recon),ignore_index=True)
#%%
short_name=['Beaver','Harrow','SRL','WSL']
all_gage['short_name']=pd.Series(short_name)

#%%
fig, ax = plt.subplots()
map = Basemap(projection='merc',resolution='i',lat_0=45,lon_0=-83, area_thresh=1000.0, llcrnrlon=min_Lon-2,llcrnrlat=min_Lat-2,urcrnrlon=max_Lon+2,urcrnrlat=max_Lat+2) #lat_0=45,lon_0=-83,
parallels = np.arange(0.,81,5.)
map.drawparallels(parallels,labels=[True,False,False,False],linewidth=0.1)  ## fontsize=ftz,
meridians = np.arange(10.,351.,5.)
map.drawmeridians(meridians,labels=[False,False,False,True],dashes=[8,1],linewidth=0.1)  ##fontsize=ftz,
map.drawcoastlines()  

X,Y=map(list(all_gage['Lon']),list(all_gage['Lat'])) 
ax.scatter(X,Y)
#for i, (x, y) in enumerate(zip(X, Y), start=0):
#    ax.annotate(all_gage['short_name'].iloc[i], (x,y), xytext=(5, 5), textcoords='offset pixels')
#    
plt.savefig('C:\\Reseach_CIGLR\\Precipitation\\station_observations\\Overlake_stations.jpg',bbox_inches='tight')
plt.show()

#%% Gage data
rain_Beaver=pd.read_csv('C:\\Reseach_CIGLR\\Precipitation\\station_observations\\OL_stations\\Precip_Day_'+str(add_gage['Stn_id'].iloc[0])+'_2010_2018.csv',sep=';',names=['Date','Precip'],skiprows=1)
rain_Harrow=pd.read_csv('C:\\Reseach_CIGLR\\Precipitation\\station_observations\\OL_stations\\Precip_Day_'+str(add_gage['Stn_id'].iloc[1])+'_2010_2018.csv',sep=';',names=['Date','Precip'],skiprows=1)
rain_SRL=pd.read_csv(os.path.join(dir_obs,'Precip_Day_huron-srl_2013_2018.csv'),delimiter=';',names=['Date','Precip'],skiprows=1)
rain_WSL=pd.read_csv(os.path.join(dir_obs,'Precip_Day_mich-wsl_2013_2018.csv'),delimiter=';',names=['Date','Precip'],skiprows=1)        

#%%
years=list(range(2014,2018))
begin_time=datetime.datetime.strptime(str(years[0])+'0101','%Y%m%d')
end_time=datetime.datetime.strptime(str(years[-1])+'1231' ,'%Y%m%d')
obs_comp=[]       
#%%
for igage in short_name:
    rain_igage=eval('rain_'+igage)
    rain_igage['Date']=pd.to_datetime(rain_igage['Date'])
    obs=rain_igage.loc[(rain_igage['Date']>=begin_time) & (rain_igage['Date']<=end_time)].reset_index()['Precip']
    obs_comp.append(obs)

#%%
for i in range(len(short_name)):
    obs_comp[i].plot(label=short_name[i])
    plt.legend()

plt.savefig('C:\\Reseach_CIGLR\\Precipitation\\station_observations\\Overlake_stations_compare.jpg',bbox_inches='tight')



#%%        
# =============================================================================
## # index_min_srl=11378, dist=0.057241
## # index_min_wsl=10546, dist=0.059593
## several minimal distance
## =============================================================================
#min2_dist_srl=nsmallest(2, dist_srl)[-1]     # 0.0664
#min3_dist_srl=nsmallest(3, dist_srl)[-1]    # 0.073928
#index_min2_srl=dist_srl.index(min2_dist_srl)   # 11170
#index_min3_srl=dist_srl.index(min3_dist_srl)  # 11169
#
#min2_dist_wsl=nsmallest(2, dist_wsl)[-1]     # 0.064001
#min3_dist_wsl=nsmallest(3, dist_wsl)[-1]    # 0.079952
#index_min2_wsl=dist_wsl.index(min2_dist_wsl)   # 10749
#index_min3_wsl=dist_wsl.index(min3_dist_wsl)  # 10750
