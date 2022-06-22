# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
# ** This script is for evaluating the the performance of different products in US and CAN
# ** Edit by Yi Hong, 07/09/2020
# ** Modified by Yi Hong, Jul. 23, 2020, plot the station missing data analysis for U.S./ CA
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

#os.environ["PROJ_LIB"] = "C:\\ProgramData\\Anaconda3\\Library\\share"; #path of the proj for basemap
#from mpl_toolkits.basemap import Basemap

import seaborn as sns

#%%
# =============================================================================
# Configuration settings
# =============================================================================
precp_product=['AORC','CaPA','MPE','Mrg']
ShpDir =r"C:\Reseach_CIGLR\Precipitation\Basin_accum\shapefiles\Download\ne_50m_admin_0_countries"      # Catchment directory

dir_stn='C:\\Reseach_CIGLR\\Precipitation\\station_observations'
dir_stn_data=os.path.join(dir_stn,'OL_stn2')
limit_na=1
stn_info_file=os.path.join(dir_stn,'anal_2010_2019\\stations_NaN_Fraction_'+str(limit_na)+'.csv')
stn_info= pd.read_csv(stn_info_file, sep=';',skipinitialspace=True)   #col_names=['Stn_id','Stn_name','Lon','Lat']

date_format='%Y%m%d'
begin_date='20100101'
end_date='20191231'

#%%
# =============================================================================
# Step 1, Input vector layer (basins, lake, land)， directly read downloaded shapefile of multipolygons
# =============================================================================
in_shp_wgs84 = gpd.read_file(os.path.join(ShpDir,'ne_50m_admin_0_countries.shp'))
shp_US=in_shp_wgs84.loc[in_shp_wgs84['NAME_EN']=='United States of America']
shp_CA=in_shp_wgs84.loc[in_shp_wgs84['NAME_EN']=='Canada']

#%%
# =============================================================================
# 2nd step, creat xarray masks
# =============================================================================
lon = stn_info['Lon']
lat = stn_info['Lat']
pnts= gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat))
#%%
fig, ax = plt.subplots()
shp_US.plot(ax=ax, facecolor='blue')
shp_CA.plot(ax=ax, facecolor='orange')
pnts.plot(ax=ax, color='red', markersize=1)
ax.set_xlim(-95, -75)
ax.set_ylim(40, 55) 
plt.title('632 GHCN stations in US/CA')
#ax.legend(loc='upper right')
plt.savefig(os.path.join(dir_stn,'anal_2010_2019\\Station_locations.jpg'))
plt.show()

#%% mask for 632 stations in US and CA
shp_US.reset_index(drop=True, inplace=True)
shp_CA.reset_index(drop=True, inplace=True)
mask_US=pnts.within(shp_US.loc[0,'geometry'])
mask_CA=pnts.within(shp_CA.loc[0,'geometry'])

#%% check if it is good
sum(mask_US) # 529, 911 in total
sum(mask_CA)  # 103, 373 in total
pnts_US=pnts.loc[mask_US]
pnts_CA=pnts.loc[mask_CA]

#%% adjusted CA stations, from the file OL_gauge
# for adjusted CA, 112 contains data after 2010, 42 have <10 % missing data for 2010 - 2015
lon_new=stn_ca_info['Longitude (Decimal Degrees)']  # stn_2010_new for 112 station, #stn_glr for 42 stations, stn_ca_info for 3346 stations
lat_new=stn_ca_info['Latitude (Decimal Degrees)']
pnts_new= gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon_new, lat_new))
#%%
fig, ax = plt.subplots()
pnts_CA.plot(ax=ax, color='k', markersize=1)
pnts_new.plot(ax=ax, color='r', markersize=1)
#%%
pnts_CA=pnts_CA.drop_duplicates()
pnts_new=pnts_new.drop_duplicates()
#%%
pnts_CA_tot=pd.concat([pnts_CA, pnts_new], axis=0)   # 215=103+112 or 485 = 373+112, or 3719 = 373+3346
sum(pnts_CA_tot.duplicated()) # 133=61+68+4
sum(pnts_CA.duplicated())  # 61 in 371 are duplicated, 19 in 103 duplicated
sum(pnts_new.duplicated())   # 68 for 3346 stns
#%%
fig, ax = plt.subplots()
pnts_US.plot(ax=ax, color='blue', markersize=1)  
pnts_CA.plot(ax=ax, color='red', markersize=1)  

#%%
# =============================================================================
# Step 3, Apply the mask and analysis
# =============================================================================
stn_US = stn_info['Stn_id'][mask_US].tolist()
stn_CA = stn_info['Stn_id'][mask_CA].tolist()
stn_country={'stn_US':stn_US,'stn_CA':stn_CA}    

#%% products
begin_time=datetime.datetime.strptime(begin_date,date_format)
end_time=datetime.datetime.strptime(end_date,date_format)
rng_date = pd.date_range(begin_time,end_time,freq='D')
rng_mon = pd.date_range(begin_time,end_time,freq='M')
precp_product=['AORC','CaPA','MPE','Mrg']
years=list(range(2010,2020))

anal_root="C:\\Reseach_CIGLR\\Precipitation\\Precip_products\\Analysis"

#%% 
list_var=['MAE','RMSE','Diff','PBias','r2']
list_country=['US','CA']

anal_total=dict([(ivar, {}) for ivar in list_var])
for ivar in list_var:
    anal_total[ivar]['US']=pd.DataFrame(columns=stn_US, index=precp_product)
    anal_total[ivar]['CA']=pd.DataFrame(columns=stn_CA, index=precp_product)

#%%
start=datetime.datetime.now()

for iprod in precp_product:
    #%
    dir_prod="C:\\Reseach_CIGLR\\Precipitation\\Precip_products\\"+iprod+"\\Precip_Land\\Data"
    anal_prod=anal_root+'\\OLand\\'+iprod
    if not os.path.isdir(anal_prod):  
        os.mkdir(anal_prod)
    #%          
    if iprod == 'HRRR': years_prod=list(range(2014,2020))
    else: years_prod=list(range(2010,2020))
    
    begin_time=datetime.datetime.strptime(str(years[0])+'0101','%Y%m%d')
    end_time=datetime.datetime.strptime(str(years[-1])+'1231','%Y%m%d')
    
    for icountry in ['US','CA']:
        stn_eval=eval('stn_'+icountry)

        for istn in stn_eval:
            #%
            rain_obs=pd.read_csv(os.path.join(dir_stn_data,'Precip_Day_'+istn+'_2010_2019.csv'),delimiter=';',names=['Date','Precip'],skiprows=1)
            rain_obs['Date']=pd.to_datetime(rain_obs['Date'])
            obs_comp=rain_obs.loc[(rain_obs['Date']>=begin_time) & (rain_obs['Date']<=end_time)].reset_index()['Precip']
            
            rain_prod=pd.read_csv(os.path.join(dir_prod, iprod+'_Day_'+str(istn)+'_'+str(years_prod[0])+'_'+str(years_prod[-1])+'.csv'),delimiter=';',names=['Date','Precip'],skiprows=1)   
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
            for i in range(len(obs_comp)):
                if obs_comp.iloc[i]<0 or obs_comp.iloc[i]>150:
                    obs_comp.iloc[i]=np.nan
            for i in range(len(prod_comp)):
                if prod_comp.iloc[i]<0 or prod_comp.iloc[i]>150:
                    prod_comp.iloc[i]=np.nan  
                    
            #% MAE, RMSE, diff, PBias, r2
            MAE, RMSE, Diff=MAE_RMSE_Diff(prod_comp, obs_comp)
            PBias=PBIAS(prod_comp, obs_comp)
            r2=R2(prod_comp, obs_comp)
            #%
            anal_total['MAE'][icountry].loc[iprod,istn]=float(MAE)        
            anal_total['RMSE'][icountry].loc[iprod,istn]=float(RMSE)         
            anal_total['Diff'][icountry].loc[iprod,istn]=float(Diff)/len(years_prod)   
            anal_total['PBias'][icountry].loc[iprod,istn]=float(PBias)
            anal_total['r2'][icountry].loc[iprod,istn]=float(r2)

print (datetime.datetime.now()-start)   
# Compute time = 24 min
 
#%% analysis and convert for boxplot
for ivar in list_var:
    #%    
    df_anal=pd.concat(value.assign(country=i_key) for i_key, value in anal_total[ivar].items())
#%
    df_convert = (
        df_anal.set_index('country', append=True)  # set product as part of the index
          .stack()                      # pull istn into rows 
          .to_frame()                   # convert to a dataframe
          .reset_index()                # make the index into reg. columns
          .rename(columns={'level_0': 'product', 0: 'value'})  # rename columns
          .drop('level_2', axis='columns')   # drop junk columns
    )
    
    df_convert['value']=df_convert['value'].astype(float)
    df_convert['country']=df_convert['country'].astype(str)
#%
    sns.set(style="ticks")    
    fig = plt.figure(figsize=(15,4))
#    colors = ["darkblue",'darkorange','darkgreen','darkred']
#    sns.set_palette(sns.color_palette(colors))
    
    ax=sns.boxplot(x="product", y="value", hue='country',data=df_convert, order=precp_product)
    
#    colors = ["darkblue",'lightsteelblue','darkorange','bisque','darkgreen','springgreen','darkred','lightcoral']
    colors = ["darkblue",'white','darkorange','white','darkgreen','white','darkred','white']
    edgecolor=["black",'darkblue','black','darkorange','black','darkgreen','black','darkred']
    
    for i,artist in enumerate(ax.artists):
        artist.set_facecolor(colors[i])
        artist.set_edgecolor(edgecolor[i])
        for j in range(i*6,i*6+6):             # Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
            line = ax.lines[j]
            line.set_color(edgecolor[i])
            line.set_mfc(edgecolor[i])
            line.set_mec(edgecolor[i])
   
    if ivar == 'PBias':
        ax.axhline(y=0, color='red', linestyle='--')
        
    ax.set_xlabel('')    
#    ax.legend(loc='lower center', ncol=3).set_title('')
    ax.legend(loc='upper right',ncol=2,mode="expand",bbox_to_anchor=(0.77, 1.1, 0.22, .102), prop={'size':16}).set_title('')
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', labelsize=18) 
    
    legpatchs = ax.get_legend().get_patches()      # also fix legend
    legpatchs[0].set_facecolor('grey')  
    legpatchs[1].set_facecolor('None')  
        
    if ivar == 'PBias':
        ax.set_ylabel('PBIAS (%)', fontsize=16, weight='bold')
        ax.set_ylim([-80,60])
    elif ivar=='MAE':
        ax.set_ylabel(ivar+' (mm/day)', fontsize=16, weight='bold')
        ax.set_ylim([0,10])
    elif ivar=='RMSE':
        ax.set_ylabel(ivar+' (mm/day)', fontsize=16, weight='bold')
        ax.set_ylim([0,10])
    elif ivar=='r2':
        ivar='R2'
        ax.set_ylabel('R2', fontsize=16, weight='bold')
        ax.set_ylim([-.4,1.1])
    else:
        ax.set_ylabel('Prod.-Obs. (mm/year)', fontsize=16, weight='bold')
#        ax.set_ylim([-350, 350])
    
    ax.set_title('Boxplots of '+ivar+' for stations in US/CA',  fontsize=20, weight='bold')    
    plt.savefig(os.path.join(anal_root,'Figures\\OLand\\'+ivar+'_US_CA_Stations.jpg'),bbox_inches='tight')
#    plt.savefig(os.path.join(anal_root,'Figures\\OLand\\'+ivar+'_entire_US_CA_Stations.eps'),bbox_inches='tight', format='eps')

    plt.show()     

#%%
# =============================================================================
# Plot missing data presentation for US, CA stations
# =============================================================================
df_na=pd.read_csv(os.path.join(dir_stn,'anal_2010_2019\\NA_OL_Stns_2010_2019.csv'),sep=';',index_col=0)
#%%
NA_tot_US=df_na.loc['Total_NA2',stn_US]
NA_tot_CA=df_na.loc['Total_NA2',stn_CA]

#%% 
fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
axs.hist(NA_tot_US, bins=100)
axs.hist(NA_tot_CA, bins=100)
axs.set_xlim(0, 0.1)
#%%  
ax1=df_na.loc['Total_NA2',stn_US].plot(kind='hist',title='Frequency distribution of missing data fraction of U.S. stations')
ax2=df_na.loc['Total_NA2',stn_CA].plot(kind='hist',title='Frequency distribution of missing data fraction of CA stations')
#%%
ax.set(xlabel='Missing_data_fraction',ylabel='Number of stations')
fig = ax.get_figure()
#fig.savefig(os.path.join(dir_stn,'Ajusted_canadian\\Missing_data_fraction.jpg'))
plt.close(fig)

# =============================================================================
# Plot monthly precipitation for US and CA GHCN
# =============================================================================
month_names=list(map(lambda x: datetime.datetime.strptime(str(x), '%m').strftime('%b'),list(range(1,13))))
rng_date = pd.period_range(begin_time,end_time,freq='D').to_timestamp()
precip_US=pd.DataFrame(columns=stn_US, index=rng_date)
precip_CA=pd.DataFrame(columns=stn_CA, index=rng_date)

#%% 
for istn in stn_US:
    rain_obs=pd.read_csv(os.path.join(dir_stn_data,'Precip_Day_'+istn+'_2010_2019.csv'),delimiter=';',index_col=0)
    rain_obs.index=pd.to_datetime(rain_obs.index)
    precip_US[istn]=rain_obs

for istn in stn_CA:
    rain_obs=pd.read_csv(os.path.join(dir_stn_data,'Precip_Day_'+istn+'_2010_2019.csv'),delimiter=';',index_col=0)
    rain_obs.index=pd.to_datetime(rain_obs.index)
    precip_CA[istn]=rain_obs

#%% 
precip_US_mon=precip_US.copy()
precip_CA_mon=precip_CA.copy()
precip_US_mon=precip_US_mon.resample("M").sum()
precip_CA_mon=precip_CA_mon.resample("M").sum()
#%%
precip_CA_mon.mean(axis=1).plot()
precip_US_mon.mean(axis=1).plot()
#%%
precip_CA_mon.index = precip_CA_mon.index.strftime('%b')
precip_US_mon.index = precip_US_mon.index.strftime('%b')
precip_US_mon['country']='US'
precip_CA_mon['country']='CA'
#%%
US_convert = (
    precip_US_mon.set_index('country', append=True)  # set product as part of the index
      .stack()                      # pull istn into rows 
      .to_frame()                   # convert to a dataframe
      .reset_index()                # make the index into reg. columns
      .rename(columns={'level_0': 'Month', 0: 'value'})  # rename columns
      .drop('level_2', axis='columns')   # drop junk columns
)

CA_convert = (
    precip_CA_mon.set_index('country', append=True)  # set product as part of the index
      .stack()                      # pull istn into rows 
      .to_frame()                   # convert to a dataframe
      .reset_index()                # make the index into reg. columns
      .rename(columns={'level_0': 'Month', 0: 'value'})  # rename columns
      .drop('level_2', axis='columns')   # drop junk columns
)
#%%
USCA_convert=pd.concat([US_convert, CA_convert], axis=0)

#%% barplots
sns.set(style="ticks")
fig = plt.figure(figsize=(15,4))
#colors = ["red", "darkblue",'darkorange','darkgreen','darkred']
#sns.set_palette(sns.color_palette(colors))
palette ={"US":"black","CA":"silver"}

ax=sns.barplot(x="Month", y="value", hue='country',data=USCA_convert,order=month_names, palette=palette,edgecolor="grey", errwidth=2) #ci=75,
ax.set_title('Monthly precipitations of US/CA stations',  fontsize=16, weight='bold') 
ax.set_ylabel('Precipitation (mm/month)', fontsize=14, weight='bold')
#ax.set_ylim([-20,20])
#    ax.axhline(0, ls='--',c='r')
ax.set_xlabel('Month', fontsize=14, weight='bold')
ax.tick_params(axis='both', labelsize=14) 
ax.legend(loc='upper left', ncol=2, fontsize=14).set_title('')
plt.savefig(os.path.join(anal_root,'Figures\\OLand\\Monthly_Precip_countries.jpg'),bbox_inches='tight')
plt.show()

#%%
# =============================================================================
# Compare with adjusted canadian stations
# =============================================================================
stn_glr=pd.read_csv(os.path.join(dir_stn,'Ajusted_canadian\\Stn_GLR_2010_2015.csv'),sep=';') 
begin_time2=datetime.datetime.strptime('20100101',date_format)
end_time2=datetime.datetime.strptime('20151231',date_format)
#rng_date2 = pd.date_range(begin_time2,end_time2,freq='D')
rng_date2 = pd.period_range(begin_time2,end_time2,freq='D').to_timestamp()
precip_CA2=pd.DataFrame(columns= stn_glr['Climate ID'].tolist(), index=rng_date2)
#%%
for iglr in stn_glr['Climate ID'].tolist():
    #
    rain_obs=pd.read_csv(os.path.join(dir_stn,'Ajusted_canadian\\Stn_GLR\\R_'+iglr+'.csv'), sep=';').drop_duplicates()  
    rain_obs['Date']=pd.to_datetime(rain_obs['Date'],format=date_format)
    rain_obs=rain_obs.set_index('Date').sort_index() 
    rain_obs = rain_obs[~rain_obs.index.duplicated()]
    precip_CA2[iglr]=rain_obs['Precip'].loc[(rain_obs.index>=begin_time2) & (rain_obs.index<=end_time2)]

#%%
precip_CA2_mon=precip_CA2.copy()
precip_US_mon=precip_US.copy()
precip_CA_mon=precip_CA.copy()
#%%
precip_US_mon=precip_US_mon.loc[(precip_US_mon.index>=begin_time2) & (precip_US_mon.index<=end_time2)]
precip_CA_mon=precip_CA.loc[(precip_CA_mon.index>=begin_time2) & (precip_CA_mon.index<=end_time2)]

#%%
precip_CA2_mon=precip_CA2_mon.resample("M").sum()
precip_US_mon=precip_US_mon.resample("M").sum()
precip_CA_mon=precip_CA_mon.resample("M").sum()

precip_CA2_mon.mean(axis=1).plot()
precip_CA_mon.mean(axis=1).plot()
precip_US_mon.mean(axis=1).plot()

#%%
precip_CA_mon.index = precip_CA_mon.index.strftime('%b')
precip_US_mon.index = precip_US_mon.index.strftime('%b')
precip_CA2_mon.index = precip_CA2_mon.index.strftime('%b')

precip_US_mon['country']='GHCN-US'
precip_CA_mon['country']='GHCN-CA'
precip_CA2_mon['country']='Adjusted-CA'
#%%
US_convert = (
    precip_US_mon.set_index('country', append=True)  # set product as part of the index
      .stack()                      # pull istn into rows 
      .to_frame()                   # convert to a dataframe
      .reset_index()                # make the index into reg. columns
      .rename(columns={'level_0': 'Month', 0: 'value'})  # rename columns
      .drop('level_2', axis='columns')   # drop junk columns
)

CA_convert = (
    precip_CA_mon.set_index('country', append=True)  # set product as part of the index
      .stack()                      # pull istn into rows 
      .to_frame()                   # convert to a dataframe
      .reset_index()                # make the index into reg. columns
      .rename(columns={'level_0': 'Month', 0: 'value'})  # rename columns
      .drop('level_2', axis='columns')   # drop junk columns
)

CA2_convert = (
    precip_CA2_mon.set_index('country', append=True)  # set product as part of the index
      .stack()                      # pull istn into rows 
      .to_frame()                   # convert to a dataframe
      .reset_index()                # make the index into reg. columns
      .rename(columns={'level_0': 'Month', 0: 'value'})  # rename columns
      .drop('level_2', axis='columns')   # drop junk columns
)

USCA2_convert=pd.concat([US_convert, CA_convert,CA2_convert], axis=0)
#%%
sns.set(style="ticks")
fig = plt.figure(figsize=(15,4))
#colors = ["red", "darkblue",'darkorange','darkgreen','darkred']
#sns.set_palette(sns.color_palette(colors))
palette ={"GHCN-US":"black","GHCN-CA":"whitesmoke","Adjusted-CA":"silver"}

ax=sns.barplot(x="Month", y="value", hue='country',data=USCA2_convert,order=month_names, palette=palette,edgecolor="grey", errwidth=2) #ci=75,
ax.set_title('Monthly precipitations of US/CA stations',  fontsize=16, weight='bold') 
ax.set_ylabel('Precipitation (mm/month)', fontsize=14, weight='bold')
#ax.set_ylim([-20,20])
#    ax.axhline(0, ls='--',c='r')
ax.set_xlabel('Month', fontsize=14, weight='bold')
ax.tick_params(axis='both', labelsize=14) 
ax.legend(loc='upper left', ncol=2, fontsize=14).set_title('')
plt.savefig(os.path.join(anal_root,'Figures\\OLand\\Monthly_Precip_datasets.jpg'),bbox_inches='tight')
plt.show()

#%% plot annual
precip_CA2_Y=precip_CA2.copy()
precip_US_Y=precip_US.copy()
precip_CA_Y=precip_CA.copy()
#%%
precip_US_Y=precip_US_Y.loc[(precip_US_Y.index>=begin_time2) & (precip_US_Y.index<=end_time2)]
precip_CA_Y=precip_CA_Y.loc[(precip_CA_Y.index>=begin_time2) & (precip_CA_Y.index<=end_time2)]

#%%
precip_CA2_Y=precip_CA2_Y.resample("Y").sum()
precip_US_Y=precip_US_Y.resample("Y").sum()
precip_CA_Y=precip_CA_Y.resample("Y").sum()

precip_CA2_Y.index = precip_CA2_Y.index.strftime('%Y')
precip_US_Y.index = precip_US_Y.index.strftime('%Y')
precip_CA_Y.index = precip_CA_Y.index.strftime('%Y')
#%%
fig,ax = plt.subplots(1, 1) #, figsize=(15,7)
precip_US_Y.mean(axis=1).plot()
precip_CA_Y.mean(axis=1).plot()
#precip_CA2_Y.mean(axis=1).plot()
#ax.legend(['GHCN-US','GHCN-CA','Adjusted-CA'])
#ax.legend(loc='lower right', ncol=3, fontsize=14).set_title('')










#%%  
# =============================================================================
#    analysis by country          
# =============================================================================
#stn_CA2=stn_glr['Climate ID'].tolist()
total_anal={}
for ivar in ['MAE','PBias']:
   total_anal[ivar]=dict([(prod, pd.DataFrame(columns=['US','CA'], index=rng_mon)) for prod in precp_product])    
#%%
for country in ['US','CA']:
    c_stn=eval('stn_'+country)
    #rng_mon = pd.period_range(begin_time,end_time,freq='M').to_timestamp()
    rng_mon = pd.date_range(begin_time,end_time,freq='M')
    mon_anal={}
    for ivar in ['MAE','PBias']:
       mon_anal[ivar]=dict([(prod, pd.DataFrame(columns=c_stn, index=rng_mon)) for prod in precp_product])    
    #%
    for iprod in precp_product:
       anal_prod=anal_root+'\\OLand\\'+iprod
       for ivar in ['MAE','PBias']:
           temp=pd.read_csv(os.path.join(anal_prod,'Monthly2_'+ivar+'_'+iprod+'.csv'),sep=';',index_col=0)
           temp.index=pd.to_datetime(temp.index,format=date_format)       
           for istn in c_stn:
               mon_anal[ivar][iprod][istn]=temp[istn]
    
           total_anal[ivar][iprod][country]=mon_anal[ivar][iprod].median(axis=1)
#%% plot line plot
for ivar in ['MAE','PBias']:
    #%%
    fig,ax = plt.subplots(1, 1) #, figsize=(15,7)
    for iprod in precp_product:
        #%%
        if iprod == "AORC": color='darkblue'
        if iprod == "CaPA": color='darkorange'
        if iprod == "MPE": color='darkgreen'
        if iprod == "Mrg": color='darkred'
        fig,ax = plt.subplots(1, 1)       
        total_anal[ivar][iprod].plot(ax=ax,y='US',lw=1, style='-',marker='s',markersize=6)# color=color,
        total_anal[ivar][iprod].plot(ax=ax, y="CA",lw=1,style='--',marker='^',markersize=6)#fillstyle='none'
    
    








    
    #%% analysis and convert for boxplot
    for ivar in ['MAE','PBias']:
        #%    
        df_anal=pd.concat(value.assign(product=i_key) for i_key, value in mon_anal[ivar].items())
        df_anal.index = df_anal.index.strftime('%b')
    #%
        df_convert = (
            df_anal.set_index('product', append=True)  # set product as part of the index
              .stack()                      # pull istn into rows 
              .to_frame()                   # convert to a dataframe
              .reset_index()                # make the index into reg. columns
              .rename(columns={'level_0': 'Month', 0: 'value'})  # rename columns
              .drop('level_2', axis='columns')   # drop junk columns
        )
        
        df_convert['value']=df_convert['value'].astype(float)
        df_convert['Month']=df_convert['Month'].astype(str)
    #%
        sns.set(style="ticks")    
        fig = plt.figure(figsize=(15,4))
        colors = ["darkblue",'darkorange','darkgreen','darkred']
        sns.set_palette(sns.color_palette(colors))
        
        ax=sns.boxplot(x="Month", y="value", hue='product',data=df_convert, order=month_names)
        ax.set_title('Boxplots of monthly '+ivar+' for '+country+' gauge stations',  fontsize=16, weight='bold')
        ax.set_xlabel('')    
    #%    ax.legend(loc='lower center', ncol=3).set_title('')
        ax.legend(loc='lower center',fontsize=14, ncol=4).set_title('')
        ax.tick_params(axis='both', labelsize=14) 
        
        if ivar == 'PBias':
            ax.axhline(y=0, color='red', linestyle='--')
            ax.set_ylabel('PBIAS (%)', fontsize=14, weight='bold')
            ax.set_ylim([-90,90])
        elif ivar=='MAE':
            ax.set_ylabel(ivar+' (mm/month)', fontsize=14, weight='bold')
            ax.set_ylim([0,6])
        elif ivar=='RMSE':
            ax.set_ylabel(ivar+' (mm/month)', fontsize=14, weight='bold')
            ax.set_ylim([0,50])
        elif ivar=='r2':
            ax.set_ylabel('R2', fontsize=14, weight='bold')
            ax.set_ylim([0.5,1.1])
        else:
            ax.set_ylabel('Prod.-Obs. (mm/month)', fontsize=14, weight='bold')
            ax.set_ylim([-50, 50])
            
        plt.savefig(os.path.join(anal_root,'Figures\\OLand\\Monthly_'+ivar+'_'+country+'.jpg'),bbox_inches='tight')
        plt.show()             
            