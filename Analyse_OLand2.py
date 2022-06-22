# This scripts is to analyze the recorded data of the overland stations
# Edited by Yi Hong, Dec. 21 2019
# Modified by Yi Hong, Mars 11 2020, add the analysis for different rainfall products
# Modified by Yi Hong, August 09, 2021, add the 10km resolution analysis

# Load the xarray package
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
from precip_functions import MAE_RMSE_Diff, PBIAS, R2, map_area

#os.environ["PROJ_LIB"] = "C:\\ProgramData\\Anaconda3\\Library\\share"; #path of the proj for basemap
#from mpl_toolkits.basemap import Basemap

import seaborn as sns

#%%
# =============================================================================
# For overland stations       
# =============================================================================
dir_stn='C:\\Reseach_CIGLR\\Precipitation\\station_observations'
dir_stn_data=os.path.join(dir_stn,'OL_stn2')
limit_na=0.1
stn_info_file=os.path.join(dir_stn,'anal_2010_2019\\stations_NaN_Fraction_'+str(limit_na)+'.csv')
stn_info= pd.read_csv(stn_info_file, sep=';',skipinitialspace=True)   #col_names=['Stn_id','Stn_name','Lon','Lat']

#%% products
#precp_product=['AORC','CaPA','MPE','Mrg','HRRR']
precp_product=['AORC','CaPA','MPE','Mrg']
years=list(range(2010,2020))

anal_root="C:\\Reseach_CIGLR\\Precipitation\\Precip_products\\Analysis"

#%% 
# =============================================================================
# Total analysis and map results
# =============================================================================
list_var=['MAE','RMSE','Diff','PBias','r2']
#% dictionary of dataframes for the total analysis results
anal_total=dict([(prod, pd.DataFrame(columns=stn_info['Stn_id'].tolist(), index=list_var)) for prod in precp_product]) 

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
    
    for istn in stn_info['Stn_id'].tolist():
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
        anal_total[iprod].loc['MAE',istn]=float(MAE)        
        anal_total[iprod].loc['RMSE',istn]=float(RMSE)         
        anal_total[iprod].loc['Diff',istn]=float(Diff)/len(years_prod)   
        anal_total[iprod].loc['PBias',istn]=float(PBias)
        anal_total[iprod].loc['r2',istn]=float(r2)
    #%    
    anal_total[iprod].to_csv(os.path.join(anal_prod,'Anal_total_'+iprod+'.csv'),sep=';')


print (datetime.datetime.now()-start)   
#%% read anal_total
for iprod in precp_product:
    #%
    dir_prod="C:\\Reseach_CIGLR\\Precipitation\\Precip_products\\"+iprod+"\\Precip_Land\\Data"
    anal_prod=anal_root+'\\OLand\\'+iprod
    anal_total[iprod]=pd.read_csv(os.path.join(anal_prod,'Anal_total_'+iprod+'.csv'),sep=';',index_col=0)

    
#%%  box plot of each variables, side by side
#mul_index=pd.MultiIndex.from_product([list(anal_total.keys()), list_var], names=['product','var'])
#df_anal=pd.DataFrame(index=mul_index, columns=stn_info['Stn_id'].tolist())
##%%
#for i_key in list(anal_total.keys()):
#    for ivar in list_var:
#        for istn in stn_info['Stn_id'].tolist():
#            df_anal.loc[(i_key, ivar), istn] = anal_total[i_key].loc[ivar, istn]
#            
#%
list_var_plot=['MAE','PBias','r2']   # only plot 3 representative metrics
#% dictionary of dataframes for the total analysis results
anal_total_plot=dict([(prod, pd.DataFrame()) for prod in precp_product]) 

for iprod in precp_product:
    anal_total_plot[iprod]=anal_total[iprod].drop(['RMSE' , 'Diff'])

#%%
df_anal_total=pd.concat(value.assign(product=i_key) for i_key, value in anal_total_plot.items())
#%
df_convert_total = (
    df_anal_total.set_index('product', append=True)  # set product as part of the index
      .stack()                      # pull istn into rows 
      .to_frame()                   # convert to a dataframe
      .reset_index()                # make the index into reg. columns
      .rename(columns={'level_0': 'var', 0: 'value'})  # rename columns
      .drop('level_2', axis='columns')   # drop junk columns
)
#%
df_convert_total['value']=df_convert_total['value'].astype(float)
df_convert_total['var']=df_convert_total['var'].astype(str)
#%%
#sns.set(style="ticks")
#colors = ["darkblue",'darkorange','darkgreen','darkred']
#sns.set_palette(sns.color_palette(colors))
#
#fig = plt.figure()
#fig.suptitle('Precipitation analysis of overland stations 2010-2019')
##plt.subplot(1, 1, 1)
#ax=sns.boxplot(x="var", y="value", hue='product',data=df_convert_total,order=['MAE','RMSE'])
#ax.set_ylabel('Metric (mm/day)', fontsize=12, weight='bold')
#ax.set_ylim([0,15])
#ax.set_xlabel('')
#ax.legend(loc='upper left').set_title('')
#plt.savefig(os.path.join(anal_root,'Figures\\OLand\\MAE_RMSE_All2.jpg'),bbox_inches='tight')
#plt.show()

#%%

fig = plt.figure(figsize=(15,4))
fig.suptitle('Precipitation analysis of 632 gauge stations over 2010-2019',fontsize=20)
plt.subplot(1, 3, 1)
colors = ["darkblue",'darkorange','darkgreen','darkred']
sns.set_palette(sns.color_palette(colors))
ax1=sns.boxplot(x="var", y="value", hue='product',data=df_convert_total,order=['MAE'])
ax1.set_xlabel('MAE (mm/day)', fontsize=14, weight='bold')
#ax1.set_ylim([0,6])
ax1.set_ylabel('')
ax1.set_xticklabels([])
ax1.tick_params(axis='both', labelsize=14)
ax1.legend().set_visible(False)
#ax1.legend(loc='lower right', ncol=5, mode="expand", borderaxespad=0).set_title('')
#ax1.set(ylabel='Metric (mm/day)')

plt.subplot(1, 3, 2)
colors = ["darkblue",'darkorange','darkgreen','darkred']
sns.set_palette(sns.color_palette(colors))
ax2=sns.boxplot(x="var", y="value", hue='product',data=df_convert_total,order=['PBias'])
#ax2.yaxis.tick_right()
#ax2.set_ylim([-40,40])
ax2.set_xlabel('PBias (%)', fontsize=14, weight='bold')
ax2.axhline(y=0, color='r', linestyle='--')
#ax2.yaxis.set_label_position("right")
ax2.set_ylabel('')
ax2.set_xticklabels([])
ax2.tick_params(axis='both', labelsize=14)
ax2.legend().set_visible(False)

plt.subplot(1, 3, 3)
colors = ["darkblue",'darkorange','darkgreen','darkred']
sns.set_palette(sns.color_palette(colors))
ax3=sns.boxplot(x="var", y="value", hue='product',data=df_convert_total,order=['r2'])
#ax3.yaxis.tick_right()
#ax3.set_ylim([-.1,1.1])
ax3.set_xlabel('R2', fontsize=14, weight='bold')
#ax3.yaxis.set_label_position("right")
ax3.set_ylabel('')
ax3.set_xticklabels([])
ax3.tick_params(axis='both', labelsize=14)
ax3.legend().set_visible(False)

ax2.legend(loc='lower left', ncol=4, mode="expand", bbox_to_anchor=(-0.5, -0.3, 2, .102),
           prop={'size':16}).set_title('')  # borderaxespad=0, [s.left, s.top+0.02, s.right-s.left, 0.05] 
#ax2.legend(loc='best', ncol=3).set_title('')
#ax2.set(ylabel='(Prod-Obs)/Obs')
#%
plt.savefig(os.path.join(anal_root,'Figures\\OLand\\Total2_MAE_PBias_R2.jpg'),bbox_inches='tight')
plt.savefig(os.path.join(anal_root,'Figures\\OLand\\Total2_MAE_PBias_R2.pdf'),bbox_inches='tight')
plt.show()

#%% map the total data
shorline_file=os.path.join(dir_stn,'greatlakes.shoreline.dat2')
min_Lon,max_Lon,min_Lat,max_Lat=map_area(shorline_file)
#%%        
v_min_max={'MAE':[0,5],'PBias':[-30,30],'r2':[0,1]}
#%
for iprod in precp_product:
    #% 
#    iprod='CaPA'
    anal_prod=anal_root+'\\OLand\\'+iprod
    df_anal=pd.read_csv(os.path.join(anal_prod,'Anal_total_'+iprod+'.csv'), sep=';',index_col=0, header=0)            
    #%
    path_fig=os.path.join(anal_root,'Figures\\OLand\\'+iprod)
    if not os.path.isdir(path_fig):  
        os.mkdir(path_fig)
    #%
    for ivar in list_var_plot:
            #%
        bmap = Basemap(projection='merc',resolution='i',lat_0=45,lon_0=-83, area_thresh=1000.0, llcrnrlon=min_Lon-2,llcrnrlat=min_Lat-2,urcrnrlon=max_Lon+2,urcrnrlat=max_Lat+2) #lat_0=45,lon_0=-83,
        parallels = np.arange(0.,81,5.)
        bmap.drawparallels(parallels,labels=[True,False,False,False],linewidth=0.1)  ## fontsize=ftz,
        meridians = np.arange(10.,351.,5.)
        bmap.drawmeridians(meridians,labels=[False,False,False,True],dashes=[8,1],linewidth=0.1)  ##fontsize=ftz,
        bmap.drawcountries()
        bmap.drawcoastlines()            
        #% add points in the figure
        if ivar in ['MAE','RMSE']:
            color_bar='RdYlGn_r'   #jet
        elif ivar in ['PBias','anual_diff']:
            color_bar='nipy_spectral'
        else:
            color_bar='RdYlGn'  #jet_r
            
        x,y=bmap(list(stn_info['Lon']),list(stn_info['Lat'])) 
        df_map=df_anal.loc[ivar]
        df_map=pd.to_numeric(df_map,errors='coerce').dropna()
        scatter = bmap.scatter(x,y,s=10,c=df_map, vmin=v_min_max[ivar][0], vmax=v_min_max[ivar][1], cmap=color_bar)
        cbar=plt.colorbar()
        #%
        
        if ivar =='MAE':      
            cbar.set_label(ivar+' (mm/day)',labelpad=10, y=0.45, rotation=270)         
        elif ivar=='PBias':
            cbar.set_label('PBias (%)',labelpad=10, y=0.45, rotation=270)
        else:
            cbar.set_label('R2 ',labelpad=10, y=0.45, rotation=270)
            ivar='R2'
        
        text_title= ivar+' for '+iprod + ' 2010-2019'      
        plt.title(text_title)
        plt.savefig(os.path.join(path_fig, iprod+'_'+ivar+'_Map_2010_2019.jpg'))
        plt.show()           
       
#%% 
# =============================================================================
#  Monthly accumulated precipitation comparisons    
# =============================================================================
# Initialisation of the dictionary of dicationaries of dataframes
begin_mon=datetime.datetime.strptime('20100101','%Y%m%d')
end_mon=datetime.datetime.strptime('20191231','%Y%m%d')
month_names=list(map(lambda x: datetime.datetime.strptime(str(x), '%m').strftime('%b'),list(range(1,13))))
#rng_mon = pd.period_range(begin_mon,end_mon,freq='M').to_timestamp()
rng_mon = pd.date_range(begin_mon,end_mon,freq='M')
all_prod = ['GHCN','AORC','CaPA','MPE','Mrg']
#%%
precip_mon=dict([(prod, pd.DataFrame(columns=stn_info['Stn_id'].tolist(), index=rng_mon)) for prod in all_prod])
#%%                 
start=datetime.datetime.now()   

for istn in stn_info['Stn_id'].tolist():
    #%       
    rain_obs=pd.read_csv(os.path.join(dir_stn_data,'Precip_Day_'+istn+'_2010_2019.csv'),delimiter=';',index_col=0)
    #% Delete outliers
    for i in range(len(rain_obs)):
        if rain_obs['Precip'].iloc[i]<0 or rain_obs['Precip'].iloc[i]>150:
            rain_obs['Precip'].iloc[i]=np.nan
    rain_obs.index=pd.to_datetime(rain_obs.index, format='%Y-%m-%d')         
    rain_obs_M=rain_obs.resample("M").sum()
    precip_mon['GHCN'][istn]=rain_obs_M  
    #%
    for iprod in precp_product:
        #%
        dir_prod="C:\\Reseach_CIGLR\\Precipitation\\Precip_products\\"+iprod+"\\Precip_Land\\Data"
        anal_prod=anal_root+'\\OLand\\'+iprod
        if not os.path.isdir(anal_prod):  
            os.mkdir(anal_prod)
        #%          
        if iprod == 'HRRR': years_prod=list(range(2014,2020))
        else: years_prod=list(range(2010,2020))   
                
        rain_prod=pd.read_csv(os.path.join(dir_prod, iprod+'_Day_'+str(istn)+'_'+str(years_prod[0])+'_'+str(years_prod[-1])+'.csv'),delimiter=';',index_col=0)   
        for i in range(len(rain_prod)):
            if rain_prod['Precip'].iloc[i]<0 or rain_prod['Precip'].iloc[i]>150:
                rain_prod['Precip'].iloc[i]=np.nan          
        rain_prod.index=pd.to_datetime(rain_prod.index, format='%Y-%m-%d')    
        rain_prod_M=rain_prod.resample("M").sum()
        precip_mon[iprod][istn]=rain_prod_M      
             
print (datetime.datetime.now()-start)                
        

#%% Plot monthly precip, sns barplot
mon_precip=dict([(prod, pd.DataFrame()) for prod in all_prod])

for iprod in all_prod:
    mon_precip[iprod]=precip_mon[iprod].copy()
    mon_precip[iprod].index = mon_precip[iprod].index.strftime('%b')
    
df_mon=pd.concat(value.assign(product=i_key) for i_key, value in mon_precip.items())
#%% Plot seasonal variations of each land, lake and products
df_convert_mon = (
    df_mon.set_index('product', append=True)  # set product as part of the index
      .stack()                      # pull istn into rows 
      .to_frame()                   # convert to a dataframe
      .reset_index()                # make the index into reg. columns
      .rename(columns={'level_0': 'Month', 0: 'value'})  # rename columns
      .drop('level_2', axis='columns')   # drop junk columns
)

df_convert_mon['value']=df_convert_mon['value'].astype(float)
df_convert_mon['product']=df_convert_mon['product'].astype(str)
df_convert_mon['Month']=df_convert_mon['Month'].astype(str)

#%%
sns.set(style="ticks")
fig = plt.figure(figsize=(15,4))
colors = ["red", "darkblue",'darkorange','darkgreen','darkred']
sns.set_palette(sns.color_palette(colors))

ax=sns.barplot(x="Month", y="value", hue='product',data=df_convert_mon,order=month_names, edgecolor="k", errwidth=2) #ci=75,
ax.set_title('Monthly precipitations at 632 gauge stations',  fontsize=16, weight='bold') 
ax.set_ylabel('Precipitation (mm/month)', fontsize=14, weight='bold')
#ax.set_ylim([-20,20])
#    ax.axhline(0, ls='--',c='r')
ax.set_xlabel('Month', fontsize=14, weight='bold')
ax.tick_params(axis='both', labelsize=14) 
ax.legend(loc='upper left', ncol=2, fontsize=14).set_title('')
plt.savefig(os.path.join(anal_root,'Figures\\OLand\\Monthly_Precip_stations.jpg'),bbox_inches='tight')
plt.show()

#%%
# =============================================================================
# Analysis for the monthly accumulated precip
# =============================================================================
# First, use the calculated precip mon in the above section, computational time is about 5 min.
mon_precip=dict([(prod, pd.DataFrame()) for prod in all_prod])
for iprod in all_prod:
    mon_precip[iprod]=precip_mon[iprod].copy()
#%% Creat one dataframe with all data, and change product name to index
df_mon=pd.concat(value.assign(product=i_key) for i_key, value in mon_precip.items())
df_precip_mon= df_mon.reset_index()
df_precip_mon = df_precip_mon.rename(columns={'index': 'Timestamp_Mon'})
df_precip_mon = df_precip_mon.set_index('product') 

#%% Compute MAE, RMSE, Diff, PBias, R2
#%%
mon_anal={} 
for ivar in list_var:
   mon_anal[ivar]=dict([(prod, pd.DataFrame(columns=stn_info['Stn_id'].tolist(), index=month_names)) for prod in precp_product])
#%%
for iprod in precp_product:
    for imonth in range(1,13):
        for istn in stn_info['Stn_id'].tolist():
            obs_comp=df_precip_mon[istn].loc[df_precip_mon['Timestamp_Mon'].dt.month==imonth].loc['GHCN']
            prod_comp=df_precip_mon[istn].loc[df_precip_mon['Timestamp_Mon'].dt.month==imonth].loc[iprod]
            #%
            MAE, RMSE, Diff=MAE_RMSE_Diff(prod_comp,obs_comp)
            Diff=Diff/(len(obs_comp)) 
            PBias=PBIAS(prod_comp,obs_comp)
            r2=R2(prod_comp,obs_comp)
            
            mon_anal['MAE'][iprod].loc[month_names[imonth-1],istn]=MAE
            mon_anal['RMSE'][iprod].loc[month_names[imonth-1],istn]=RMSE
            mon_anal['Diff'][iprod].loc[month_names[imonth-1],istn]=Diff
            mon_anal['PBias'][iprod].loc[month_names[imonth-1],istn]=PBias
            mon_anal['r2'][iprod].loc[month_names[imonth-1],istn]=r2

#%% analysis and convert for boxplot
for ivar in list_var:
    #%    
    df_anal=pd.concat(value.assign(product=i_key) for i_key, value in mon_anal[ivar].items())
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
    ax.set_title('Boxplots of monthly '+ivar+' for 632 gauge stations',  fontsize=16, weight='bold')
    ax.set_xlabel('')    
#    ax.legend(loc='lower center', ncol=3).set_title('')
    ax.legend(loc='lower center',fontsize=14, ncol=4).set_title('')
    ax.tick_params(axis='both', labelsize=14) 
    
    if ivar == 'PBias':
        ax.set_ylabel('PBIAS (%)', fontsize=14, weight='bold')
        ax.set_ylim([-70,70])
    elif ivar=='MAE':
        ax.set_ylabel(ivar+' (mm/month)', fontsize=14, weight='bold')
        ax.set_ylim([0,50])
    elif ivar=='RMSE':
        ax.set_ylabel(ivar+' (mm/month)', fontsize=14, weight='bold')
        ax.set_ylim([0,50])
    elif ivar=='r2':
        ax.set_ylabel('R2', fontsize=14, weight='bold')
        ax.set_ylim([0.5,1.1])
    else:
        ax.set_ylabel('Prod.-Obs. (mm/month)', fontsize=14, weight='bold')
        ax.set_ylim([-50, 50])
        
    plt.savefig(os.path.join(anal_root,'Figures\\OLand\\Monthly_'+ivar+'_All_Stations.jpg'),bbox_inches='tight')
    plt.show()
#%%
# =============================================================================
# Calculate every day in month.
# Since there are NaNs, monthly accumulated sometimes not well represented
# =============================================================================
start=datetime.datetime.now()   

mon_anal={} 
for ivar in list_var:
   mon_anal[ivar]=dict([(prod, pd.DataFrame(columns=stn_info['Stn_id'].tolist(), index=month_names)) for prod in precp_product])
   
for iprod in precp_product:
    #%
    dir_prod="C:\\Reseach_CIGLR\\Precipitation\\Precip_products\\"+iprod+"\\Precip_Land\\Data"
    anal_prod=anal_root+'\\OLand\\'+iprod
    if not os.path.isdir(anal_prod):  
        os.mkdir(anal_prod)
    #%          
    if iprod == 'HRRR': years_prod=list(range(2014,2020))
    else: years_prod=list(range(2010,2020))   
   
    #%
    for istn in stn_info['Stn_id'].tolist():
        #%        
        rain_obs=pd.read_csv(os.path.join(dir_stn_data,'Precip_Day_'+istn+'_2010_2019.csv'),delimiter=';',names=['Date','Precip'],skiprows=1)
        rain_obs['Date']=pd.to_datetime(rain_obs['Date'])
         
        rain_prod=pd.read_csv(os.path.join(dir_prod, iprod+'_Day_'+str(istn)+'_'+str(years_prod[0])+'_'+str(years_prod[-1])+'.csv'),delimiter=';',names=['Date','Precip'],skiprows=1)   
        rain_prod['Date']=pd.to_datetime(rain_prod['Date'])        
        #%
        for imonth in range(1,13):
            obs_comp=pd.Series([])
            prod_comp=pd.Series([])  #prod_comp=dict([(prod, pd.Series([])) for prod in product])
            number_days=calendar.monthrange(2019, imonth)[1]
            #%
            for iyear in years:
                #%
                begin_time=datetime.datetime.strptime(str(iyear)+'%02d' %imonth+'01','%Y%m%d')
                end_time=datetime.datetime.strptime(str(iyear)+'%02d' %imonth+ str(calendar.monthrange(iyear, imonth)[1]) ,'%Y%m%d')
                obs_mon=rain_obs.loc[(rain_obs['Date']>=begin_time) & (rain_obs['Date']<=end_time)].reset_index()['Precip']
                obs_comp=obs_comp.append(obs_mon)                             
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
                
                prod_comp=prod_comp.append(prod_comp_mon)
            #% Delete outliers
            for i in range(len(obs_comp)):
                if obs_comp.iloc[i]<0 or obs_comp.iloc[i]>150:
                    obs_comp.iloc[i]=np.nan 
            for i in range(len(prod_comp)):
                if prod_comp.iloc[i]<0 or prod_comp.iloc[i]>150:
                    prod_comp.iloc[i]=np.nan  
        #% After calculation, calculate the total monthly analysis
            MAE, RMSE, Diff=MAE_RMSE_Diff(prod_comp,obs_comp)
            mon_count=(prod_comp.reset_index(drop=True)-obs_comp.reset_index(drop=True)).count()
            Diff=Diff/(mon_count/number_days) 
            PBias=PBIAS(prod_comp,obs_comp)
            r2=R2(prod_comp,obs_comp)
            #%
            mon_anal['MAE'][iprod].loc[month_names[imonth-1],istn]=MAE
            mon_anal['RMSE'][iprod].loc[month_names[imonth-1],istn]=RMSE
            mon_anal['Diff'][iprod].loc[month_names[imonth-1],istn]=Diff
            mon_anal['PBias'][iprod].loc[month_names[imonth-1],istn]=PBias
            mon_anal['r2'][iprod].loc[month_names[imonth-1],istn]=r2
    #% record analysis results
    for ivar in list_var:
        mon_anal[ivar][iprod].to_csv(os.path.join(anal_prod,'Monthly_'+ivar+'_'+iprod+'.csv'),sep=';')


print (datetime.datetime.now()-start)  

#%% read mon_anal
mon_anal={}
for ivar in list_var:
   mon_anal[ivar]=dict([(prod, pd.DataFrame(columns=stn_info['Stn_id'].tolist(), index=month_names)) for prod in precp_product])
   

for iprod in precp_product:
    dir_prod="C:\\Reseach_CIGLR\\Precipitation\\Precip_products\\"+iprod+"\\Precip_Land\\Data"
    anal_prod=anal_root+'\\OLand\\'+iprod
    for ivar in list_var:
        mon_anal[ivar][iprod]=pd.read_csv(os.path.join(anal_prod,'Monthly_'+ivar+'_'+iprod+'.csv'),sep=';',index_col=0)



#%% analysis and convert for boxplot
for ivar in list_var:
    #%    
    df_anal=pd.concat(value.assign(product=i_key) for i_key, value in mon_anal[ivar].items())
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
    if ivar == 'PBias':
        ax.axhline(y=0, color='red', linestyle='--')
        
    ax.set_title('Boxplots of monthly '+ivar+' for 632 gauge stations',  fontsize=16, weight='bold')
    ax.set_xlabel('')    
#    ax.legend(loc='lower center', ncol=3).set_title('')
#    ax.legend(loc='upper center',fontsize=14, ncol=4).set_title('')
    
    
    if ivar == 'PBias':
        ax.set_ylabel('PBIAS (%)', fontsize=14, weight='bold')
        ax.set_ylim([-80,100])
        ax.legend(loc='lower center',fontsize=14, ncol=4).set_title('')
    elif ivar=='MAE':
        ax.set_ylabel(ivar+' (mm/day)', fontsize=14, weight='bold')
        ax.set_ylim([0,9])
        ax.legend(loc='upper left',fontsize=14, ncol=4).set_title('')
    elif ivar=='RMSE':
        ax.set_ylabel(ivar+' (mm/day)', fontsize=14, weight='bold')
        ax.set_ylim([0,10])
    elif ivar=='r2':
        ax.set_ylabel('R2', fontsize=14, weight='bold')
        ax.set_ylim([-.1,1.1])
    else:
        ax.set_ylabel('Prod.-Obs. (mm/month)', fontsize=14, weight='bold')
        ax.set_ylim([-50, 50])
     
    ax.tick_params(axis='both', labelsize=14)  
    plt.savefig(os.path.join(anal_root,'Figures\\OLand\\Monthly2_'+ivar+'_All_Stations.jpg'),bbox_inches='tight')
#    plt.savefig(os.path.join(anal_root,'Figures\\OLand\\Monthly2_entire_'+ivar+'_All_Stations.pdf'),bbox_inches='tight')
    plt.show()     

#%%
# =============================================================================
# Calculate daily MAE, R2, PBias, etc, of month, then plot for the entire period
# Since there are NaNs, monthly accumulated sometimes not well represented
# =============================================================================
start=datetime.datetime.now()  # 40 min 

mon_anal={} 
for ivar in list_var:
   mon_anal[ivar]=dict([(prod, pd.DataFrame(columns=stn_info['Stn_id'].tolist(), index=rng_mon)) for prod in precp_product])
   
for iprod in precp_product:
    #%
    dir_prod="C:\\Reseach_CIGLR\\Precipitation\\Precip_products\\"+iprod+"\\Precip_Land\\Data"
    anal_prod=anal_root+'\\OLand\\'+iprod
    if not os.path.isdir(anal_prod):  
        os.mkdir(anal_prod)
    #%          
    if iprod == 'HRRR': years_prod=list(range(2014,2020))
    else: years_prod=list(range(2010,2020))   
   
    #%
    for istn in stn_info['Stn_id'].tolist():
        #% start=datetime.datetime.now()                  
        rain_obs=pd.read_csv(os.path.join(dir_stn_data,'Precip_Day_'+istn+'_2010_2019.csv'),delimiter=';',names=['Date','Precip'],skiprows=1)
        rain_obs['Date']=pd.to_datetime(rain_obs['Date'])
         
        rain_prod=pd.read_csv(os.path.join(dir_prod, iprod+'_Day_'+str(istn)+'_'+str(years_prod[0])+'_'+str(years_prod[-1])+'.csv'),delimiter=';',names=['Date','Precip'],skiprows=1)   
        rain_prod['Date']=pd.to_datetime(rain_prod['Date'])        
        #%
        for iyear in years:
            for imonth in range(1,13):
                number_days=calendar.monthrange(iyear, imonth)[1]
                c_timeindex=datetime.datetime.strptime(str(iyear)+'%02d' %imonth+str(number_days),'%Y%m%d')
            #%
                begin_time=datetime.datetime.strptime(str(iyear)+'%02d' %imonth+'01','%Y%m%d')
                end_time=datetime.datetime.strptime(str(iyear)+'%02d' %imonth+ str(calendar.monthrange(iyear, imonth)[1]) ,'%Y%m%d')
                obs_comp=rain_obs.loc[(rain_obs['Date']>=begin_time) & (rain_obs['Date']<=end_time)].reset_index()['Precip']                           
                #%
                if (rain_prod['Date'].iloc[0]<=begin_time) & (rain_prod['Date'].iloc[-1]>=end_time):
                    prod_comp=rain_prod.loc[(rain_prod['Date']>=begin_time) & (rain_prod['Date']<=end_time)].reset_index()['Precip']
                else:
                    compare_date = pd.date_range(begin_time,end_time,freq='D')
                    prod_comp=pd.DataFrame(index=compare_date, columns=['Precip'])                    
                    temp_prod=rain_prod[rain_prod['Date'].isin(compare_date)]
                    temp_prod=temp_prod.set_index('Date')
                    prod_comp['Precip']=temp_prod['Precip']

                #% Delete outliers
                for i in range(len(obs_comp)):
                    if obs_comp.iloc[i]<0 or obs_comp.iloc[i]>150:
                        obs_comp.iloc[i]=np.nan 
                for i in range(len(prod_comp)):
                    if prod_comp.iloc[i]<0 or prod_comp.iloc[i]>150:
                        prod_comp.iloc[i]=np.nan  
                
                #% After, calculate the monthly analysis
                MAE, RMSE, Diff=MAE_RMSE_Diff(prod_comp,obs_comp)
                Diff=Diff/number_days
                PBias=PBIAS(prod_comp,obs_comp)
                r2=R2(prod_comp,obs_comp)
                #%
                mon_anal['MAE'][iprod].loc[c_timeindex,istn]=MAE
                mon_anal['RMSE'][iprod].loc[c_timeindex,istn]=RMSE
                mon_anal['Diff'][iprod].loc[c_timeindex,istn]=Diff
                mon_anal['PBias'][iprod].loc[c_timeindex,istn]=PBias
                mon_anal['r2'][iprod].loc[c_timeindex,istn]=r2
    #% record analysis results
    for ivar in list_var:
        mon_anal[ivar][iprod].to_csv(os.path.join(anal_prod,'Monthly2_'+ivar+'_'+iprod+'.csv'),sep=';',date_format='%Y%m%d')


print (datetime.datetime.now()-start)  

#%% analysis and convert for boxplot
for ivar in list_var:
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
    if ivar == 'PBias':
        ax.axhline(y=0, color='red', linestyle='--')
        
    ax.set_xlabel('')    
#    ax.legend(loc='lower center', ncol=3).set_title('')
    ax.legend(loc='lower center',fontsize=14, ncol=4).set_title('')
    ax.tick_params(axis='both', labelsize=14) 
    
    if ivar == 'PBias':
        ax.set_ylabel('PBIAS (%)', fontsize=14, weight='bold')
        ax.set_ylim([-80,80])
    elif ivar=='MAE':
        ax.set_ylabel(ivar+' (mm/day)', fontsize=14, weight='bold')
        ax.set_ylim([0,5])
    elif ivar=='RMSE':
        ax.set_ylabel(ivar+' (mm/day)', fontsize=14, weight='bold')
        ax.set_ylim([0,10])
    elif ivar=='r2':
        ivar='R2'
        ax.set_ylabel('R2', fontsize=14, weight='bold')
        ax.set_ylim([-.1,1.1])
    else:
        ax.set_ylabel('Prod.-Obs. (mm/day)', fontsize=14, weight='bold')
        ax.set_ylim([-2, 2])
     
    ax.set_title('Boxplots of monthly '+ivar+' for 632 gauge stations',  fontsize=16, weight='bold')    
    plt.savefig(os.path.join(anal_root,'Figures\\OLand\\Monthly3_'+ivar+'_All_Stations.jpg'),bbox_inches='tight')
    plt.show()     

