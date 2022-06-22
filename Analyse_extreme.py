# This scripts is to analyze the precipitation extremes
# Edited by Yi Hong, Oct. 06 2021

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
dir_stn='C:\\Research_Mich\\Precipitation\\station_observations'
dir_stn_data=os.path.join(dir_stn,'OL_stn2')
limit_na=0.1
stn_info_file=os.path.join(dir_stn,'anal_2010_2019\\stations_NaN_Fraction_'+str(limit_na)+'.csv')
stn_info= pd.read_csv(stn_info_file, sep=';',skipinitialspace=True)   #col_names=['Stn_id','Stn_name','Lon','Lat']

#%% products
# precp_product=['AORC','CaPA','MPE','Mrg','HRRR']
precp_product=['AORC','AORC10','CaPA','MPE','MPE10','Mrg']
years=list(range(2010,2020))

anal_root="C:\\Research_Mich\\Precipitation\\Precip_products\\Analysis"

#%%
# =============================================================================
# read observed precipitation and calculate percentile
# =============================================================================
begin_time=datetime.datetime.strptime(str(years[0])+'0101','%Y%m%d')
end_time=datetime.datetime.strptime(str(years[-1])+'1231','%Y%m%d')
threashold_precip=1     # select only > 1mm/day precipitations
all_obs= pd.Series([])

for istn in stn_info['Stn_id'].tolist():
    #%
    rain_obs=pd.read_csv(os.path.join(dir_stn_data,'Precip_Day_'+istn+'_2010_2019.csv'),delimiter=';',names=['Date','Precip'],skiprows=1)
    rain_obs['Date']=pd.to_datetime(rain_obs['Date'])
    obs_comp=rain_obs.loc[(rain_obs['Date']>=begin_time) & (rain_obs['Date']<=end_time)].reset_index()['Precip']
    all_obs = pd.concat([all_obs, obs_comp.loc[obs_comp >= threashold_precip ]], ignore_index=True)
#%
# all_obs.plot(kind='box')

#%% calculate the quantiles for all the products
all_quantil=pd.DataFrame(index=['0.99','0.95','0.90','0.75','0.5'], columns=['Observation']+precp_product)
all_quantil.loc['0.99','Observation']=all_obs.quantile(0.99)  # 48.3 mm/day
all_quantil.loc['0.95','Observation']=all_obs.quantile(0.95)  # 26.9 mm/day
all_quantil.loc['0.90','Observation']=all_obs.quantile(0.9)  # 19.6mm/day
all_quantil.loc['0.75','Observation']=all_obs.quantile(0.75)  # 10.5
all_quantil.loc['0.5','Observation']=all_obs.quantile(0.5)  # 4.8

#%%
years_prod=years

for iprod in precp_product:
    #%
    dir_prod="C:\\Research_Mich\\Precipitation\\Precip_products\\"+iprod+"\\Precip_Land\\Data"
    all_prod=pd.Series([])
    
    for istn in stn_info['Stn_id'].tolist():
   
        if iprod == 'AORC10':
            rain_prod=pd.read_csv(os.path.join(dir_prod, 'AORC_Day_'+str(istn)+'_'+str(years_prod[0])+'_'+str(years_prod[-1])+'.csv'),delimiter=';',names=['Date','Precip'],skiprows=1,index_col=0)
            # rain_prod['Date']=pd.to_datetime(rain_prod['Date'],format='%Y%m%d')
        elif iprod == 'MPE10':
            rain_prod=pd.read_csv(os.path.join(dir_prod, 'MPE_Day_'+str(istn)+'_'+str(years_prod[0])+'_'+str(years_prod[-1])+'.csv'),delimiter=';',names=['Date','Precip'],skiprows=1,index_col=0)
            # rain_prod['Date']=pd.to_datetime(rain_prod['Date'],format='%Y%m%d')
        else:
            rain_prod=pd.read_csv(os.path.join(dir_prod, iprod+'_Day_'+str(istn)+'_'+str(years_prod[0])+'_'+str(years_prod[-1])+'.csv'),delimiter=';',names=['Date','Precip'],skiprows=1)   
            # rain_prod['Date']=pd.to_datetime(rain_prod['Date'])
        prod_comp=rain_prod['Precip']              
        all_prod = pd.concat([all_prod, prod_comp.loc[prod_comp >= threashold_precip ]], ignore_index=True)

    all_quantil.loc['0.99',iprod]=all_prod.quantile(0.99)  # 48.3 mm/day
    all_quantil.loc['0.95',iprod]=all_prod.quantile(0.95)  # 26.9 mm/day
    all_quantil.loc['0.90',iprod]=all_prod.quantile(0.9)  # 19.6mm/day
    all_quantil.loc['0.75',iprod]=all_prod.quantile(0.75)  # 10.5
    all_quantil.loc['0.5',iprod]=all_prod.quantile(0.5)  # 4.8


all_quantil.to_csv(os.path.join(anal_root,'OLand\\Quantils_extreme.csv'),sep=',')

#%%
# =============================================================================
# Analysis only for the extreme events of different products
# =============================================================================
# limit_extrem=26.9   # from previous study, extreme event threshold is 26.9 mm/day

list_var=['MAE','RMSE','Diff','PBias','r2']
#% dictionary of dataframes for the total analysis results
anal_total=dict([(prod, pd.DataFrame(columns=stn_info['Stn_id'].tolist(), index=list_var)) for prod in precp_product]) 

df_extrem=pd.DataFrame(columns=stn_info['Stn_id'].tolist(), index=['0.99','0.95','0.75','0.5'])
#%%
for iprod in precp_product:
    #%
    dir_prod="C:\\Research_Mich\\Precipitation\\Precip_products\\"+iprod+"\\Precip_Land\\Data"
    anal_prod=anal_root+'\\OLand\\'+iprod
    if not os.path.isdir(anal_prod):  
        os.mkdir(anal_prod)
    
    for istn in stn_info['Stn_id'].tolist():
        #%
        rain_obs=pd.read_csv(os.path.join(dir_stn_data,'Precip_Day_'+istn+'_2010_2019.csv'),delimiter=';',names=['Date','Precip'],skiprows=1)
        rain_obs['Date']=pd.to_datetime(rain_obs['Date'])
        obs_temp=rain_obs.loc[(rain_obs['Date']>=begin_time) & (rain_obs['Date']<=end_time)]
        limit_extrem=obs_temp.quantile(0.95)  # precip extrem for each station
        df_extrem.loc[istn,'0.99']=float(obs_temp.quantile(0.99).values)
        df_extrem.loc[istn,'0.95']=float(obs_temp.quantile(0.95).values)
        df_extrem.loc[istn,'0.75']=float(obs_temp.quantile(0.75).values)
        df_extrem.loc[istn,'0.5']=float(obs_temp.quantile(0.5).values)
        
        obs_extrem=obs_temp.loc[(obs_temp['Precip']>=float(limit_extrem.values))]
        obs_comp=obs_extrem.reset_index()['Precip']
        #%
        if iprod == 'AORC10':
            rain_prod=pd.read_csv(os.path.join(dir_prod, 'AORC_Day_'+str(istn)+'_'+str(years_prod[0])+'_'+str(years_prod[-1])+'.csv'),delimiter=';',names=['Date','Precip'],skiprows=1,index_col=0)
            rain_prod['Date']=pd.to_datetime(rain_prod['Date'],format='%Y%m%d')
        elif iprod == 'MPE10':
            rain_prod=pd.read_csv(os.path.join(dir_prod, 'MPE_Day_'+str(istn)+'_'+str(years_prod[0])+'_'+str(years_prod[-1])+'.csv'),delimiter=';',names=['Date','Precip'],skiprows=1,index_col=0)
            rain_prod['Date']=pd.to_datetime(rain_prod['Date'],format='%Y%m%d')
        else:
            rain_prod=pd.read_csv(os.path.join(dir_prod, iprod+'_Day_'+str(istn)+'_'+str(years_prod[0])+'_'+str(years_prod[-1])+'.csv'),delimiter=';',names=['Date','Precip'],skiprows=1)   
            rain_prod['Date']=pd.to_datetime(rain_prod['Date'])

        #%                                
        compare_date = obs_extrem['Date']
        temp_prod=rain_prod[rain_prod['Date'].isin(compare_date)]
        prod_comp=temp_prod.reset_index()['Precip']
        
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
    anal_total[iprod].to_csv(os.path.join(anal_prod,'Anal_extreme_'+iprod+'.csv'),sep=',')

df_extrem.to_csv(os.path.join(anal_root,'OLand\\Quantils_extreme_stations.csv'),sep=',')
      
#%% read anal_total
for iprod in precp_product:
    #%
    dir_prod="C:\\Research_Mich\\Precipitation\\Precip_products\\"+iprod+"\\Precip_Land\\Data"
    anal_prod=anal_root+'\\OLand\\'+iprod
    anal_total[iprod]=pd.read_csv(os.path.join(anal_prod,'Anal_extreme_'+iprod+'.csv'),sep=';',index_col=0)

# =============================================================================
# Boxplot only for days of extreme precipitations, all stations
# =============================================================================
#%%
list_var_plot=['MAE','PBias','r2']   # only plot 3 representative metrics
df_anal_all=pd.concat(value.assign(product=i_key) for i_key, value in anal_total.items())
#%
df_convert_all = (
    df_anal_all.set_index('product', append=True)  # set product as part of the index
      .stack()                      # pull istn into rows 
      .to_frame()                   # convert to a dataframe
      .reset_index()                # make the index into reg. columns
      .rename(columns={'level_0': 'var', 0: 'value'})  # rename columns
      .drop('level_2', axis='columns')   # drop junk columns
)
#%
df_convert_all['value']=df_convert_all['value'].astype(float)
df_convert_all['var']=df_convert_all['var'].astype(str)


#%%
fig = plt.figure(figsize=(15,4))
fig.suptitle('Precipitation analysis for heavy rains',fontsize=20)
plt.subplot(1, 3, 1)
colors = ["darkblue","lightblue",'darkorange','darkgreen',"lightgreen",'darkred']
sns.set_palette(sns.color_palette(colors))
ax1=sns.boxplot(x="var", y="value", hue='product',data=df_convert_all,order=['MAE'])
ax1.set_xlabel('MAE (mm/day)', fontsize=14, weight='bold')
ax1.set_ylim([0,50])
ax1.set_ylabel('')
ax1.set_xticklabels([])
ax1.tick_params(axis='both', labelsize=14)
ax1.legend().set_visible(False)
#ax1.legend(loc='lower right', ncol=5, mode="expand", borderaxespad=0).set_title('')
#ax1.set(ylabel='Metric (mm/day)')

plt.subplot(1, 3, 2)
colors = ["darkblue","lightblue",'darkorange','darkgreen',"lightgreen",'darkred']
sns.set_palette(sns.color_palette(colors))
ax2=sns.boxplot(x="var", y="value", hue='product',data=df_convert_all,order=['PBias'])
#ax2.yaxis.tick_right()
ax2.set_ylim([-100,80])
ax2.set_xlabel('PBias (%)', fontsize=14, weight='bold')
ax2.axhline(y=0, color='r', linestyle='--')
#ax2.yaxis.set_label_position("right")
ax2.set_ylabel('')
ax2.set_xticklabels([])
ax2.tick_params(axis='both', labelsize=14)
ax2.legend().set_visible(False)

plt.subplot(1, 3, 3)
colors = ["darkblue","lightblue",'darkorange','darkgreen',"lightgreen",'darkred']
sns.set_palette(sns.color_palette(colors))
ax3=sns.boxplot(x="var", y="value", hue='product',data=df_convert_all,order=['r2'])
#ax3.yaxis.tick_right()
#ax3.set_ylim([-.1,1.1])
ax3.set_xlabel('R2', fontsize=14, weight='bold')
#ax3.yaxis.set_label_position("right")
ax3.set_ylabel('')
ax3.set_xticklabels([])
ax3.tick_params(axis='both', labelsize=14)
ax3.legend().set_visible(False)

ax2.legend(loc='lower left', ncol=3, mode="expand", bbox_to_anchor=(-0.5, -0.4, 2, .102),
           prop={'size':16}).set_title('')  # borderaxespad=0, [s.left, s.top+0.02, s.right-s.left, 0.05] 
#ax2.legend(loc='best', ncol=3).set_title('')
#ax2.set(ylabel='(Prod-Obs)/Obs')
#%
plt.savefig(os.path.join(anal_root,'Figures\\OLand\\Extreme_All_MAE_PBias_R2.jpg'),bbox_inches='tight')
# plt.savefig(os.path.join(anal_root,'Figures\\OLand\\Total3_MAE_PBias_R2.pdf'),bbox_inches='tight')
plt.show()

#%% It is not a good idea to include all the 1km-10km products for extremes rains
# Use only AORC, CaPA, MPE, Mrg for plot

anal_plot=dict((k, anal_total[k]) for k in ('AORC','CaPA','MPE','Mrg'))
df_anal_total=pd.concat(value.assign(product=i_key) for i_key, value in anal_plot.items())
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


fig = plt.figure(figsize=(15,4))
fig.suptitle('Precipitation analysis for heavy rains',fontsize=20)
plt.subplot(1, 3, 1)
colors = ["darkblue",'darkorange','darkgreen','darkred']
sns.set_palette(sns.color_palette(colors))
ax1=sns.boxplot(x="var", y="value", hue='product',data=df_convert_total,order=['MAE'])
ax1.set_xlabel('MAE (mm/day)', fontsize=14, weight='bold')
ax1.set_ylim([0,50])
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
ax2.set_ylim([-100,80])
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
ax3.set_ylim([-.33,1])
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
plt.savefig(os.path.join(anal_root,'Figures\\OLand\\Extreme_origin_MAE_PBias_R2.jpg'),bbox_inches='tight')
# plt.savefig(os.path.join(anal_root,'Figures\\OLand\\Total2_MAE_PBias_R2.pdf'),bbox_inches='tight')
plt.show()

#%%
# =============================================================================
# For 10km-resolution, comparisons with observations not good
# Perform direct comparisons AORC-AORC10, MPE-MPE10, with selected stations

bol_AORC10=pd.read_csv(os.path.join(anal_root+'\\OLand\\AORC10\\Diff_AORC10_bol.csv'),index_col=0,sep=',')
bol_MPE10=pd.read_csv(os.path.join(anal_root+'\\OLand\\MPE10\\Diff_MPE10_bol.csv'),index_col=0,sep=',')
#%
stn_list_AORC10=bol_AORC10.index[bol_AORC10['Pb10'] == True].tolist()
stn_list_MPE10=bol_MPE10.index[bol_MPE10['Pb25'] == True].tolist()
#%% Now, calculate the PBias between 1-10km resolution only for selected stations and compare days
dir_AORC10="C:\\Research_Mich\\Precipitation\\Precip_products\\AORC10\\Precip_Land\\Data"
dir_AORC="C:\\Research_Mich\\Precipitation\\Precip_products\\AORC\\Precip_Land\\Data"
dir_MPE10="C:\\Research_Mich\\Precipitation\\Precip_products\\MPE10\\Precip_Land\\Data"
dir_MPE="C:\\Research_Mich\\Precipitation\\Precip_products\\MPE\\Precip_Land\\Data"

#%%
diff_extrem_res10=dict([(prod, pd.DataFrame(columns=stn_info['Stn_id'].tolist(), index=['PBias','r2'])) for prod in ['AORC10','MPE10']]) 

for istn in stn_info['Stn_id'].tolist():   
    #%
    rain_obs=pd.read_csv(os.path.join(dir_stn_data,'Precip_Day_'+istn+'_2010_2019.csv'),delimiter=';',names=['Date','Precip'],skiprows=1)
    rain_obs['Date']=pd.to_datetime(rain_obs['Date'])
    obs_temp=rain_obs.loc[(rain_obs['Date']>=begin_time) & (rain_obs['Date']<=end_time)]
    limit_extrem=obs_temp.quantile(0.95)  # precip extrem for each station
    obs_extrem=obs_temp.loc[(obs_temp['Precip']>=float(limit_extrem.values))]
    obs_comp=obs_extrem.reset_index()['Precip']
    compare_date = obs_extrem['Date'] 

    rain_AORC10=pd.read_csv(os.path.join(dir_AORC10, 'AORC_Day_'+str(istn)+'_'+str(years_prod[0])+'_'+str(years_prod[-1])+'.csv'),delimiter=';',names=['Date','Precip'],skiprows=1,index_col=0)
    rain_AORC10['Date']=pd.to_datetime(rain_AORC10['Date'],format='%Y%m%d')
    temp_AORC10=rain_AORC10[rain_AORC10['Date'].isin(compare_date)]
    AORC10_comp=temp_AORC10.reset_index()['Precip']
    
    rain_AORC=pd.read_csv(os.path.join(dir_AORC, 'AORC_Day_'+str(istn)+'_'+str(years_prod[0])+'_'+str(years_prod[-1])+'.csv'),delimiter=';',names=['Date','Precip'],skiprows=1)   
    rain_AORC['Date']=pd.to_datetime(rain_AORC['Date'])
    temp_AORC=rain_AORC[rain_AORC['Date'].isin(compare_date)]
    AORC_comp=temp_AORC.reset_index()['Precip']
    
    rain_MPE10=pd.read_csv(os.path.join(dir_MPE10, 'MPE_Day_'+str(istn)+'_'+str(years_prod[0])+'_'+str(years_prod[-1])+'.csv'),delimiter=';',names=['Date','Precip'],skiprows=1,index_col=0)
    rain_MPE10['Date']=pd.to_datetime(rain_MPE10['Date'],format='%Y%m%d')
    temp_MPE10=rain_MPE10[rain_MPE10['Date'].isin(compare_date)]
    MPE10_comp=temp_MPE10.reset_index()['Precip']
    
    rain_MPE=pd.read_csv(os.path.join(dir_MPE, 'MPE_Day_'+str(istn)+'_'+str(years_prod[0])+'_'+str(years_prod[-1])+'.csv'),delimiter=';',names=['Date','Precip'],skiprows=1)   
    rain_MPE['Date']=pd.to_datetime(rain_MPE['Date'])
    temp_MPE=rain_MPE[rain_MPE['Date'].isin(compare_date)]
    MPE_comp=temp_MPE.reset_index()['Precip']    

    #% MAE, RMSE, diff, PBias, r2
    PBias_AORC=PBIAS(AORC_comp, AORC10_comp)
    r2_AORC=R2(AORC_comp, AORC10_comp)
    PBias_MPE=PBIAS(MPE_comp, MPE10_comp)
    r2_MPE=R2(MPE_comp, MPE10_comp)

    diff_extrem_res10['AORC10'].loc['PBias',istn]=float(PBias_AORC)
    diff_extrem_res10['AORC10'].loc['r2',istn]=float(r2_AORC)
    diff_extrem_res10['MPE10'].loc['PBias',istn]=float(PBias_MPE)
    diff_extrem_res10['MPE10'].loc['r2',istn]=float(r2_MPE)
#%
diff_extrem_res10['AORC10'].to_csv(os.path.join(anal_root+'\\OLand\\AORC10\\Diff_Extreme_AORC10.csv'),sep=',')
diff_extrem_res10['MPE10'].to_csv(os.path.join(anal_root+'\\OLand\\MPE10\\Diff_Extreme_MPE10.csv'),sep=',')

#%% select data and plot

Pbias_AORC10=diff_extrem_res10['AORC10'].loc['PBias',stn_list_AORC10]
Pbias_MPE10=diff_extrem_res10['MPE10'].loc['PBias',stn_list_MPE10]

#%%    
# comp_df=pd.concat(value.assign(product=i_key) for i_key, value in diff_res10.items())
comp_df = pd.concat([Pbias_AORC10, Pbias_MPE10], axis=1)
# comp_df = pd.concat([diff_res10['AORC10'].loc['PBias',:], diff_res10['MPE10'].loc['PBias',:]], axis=1)
comp_df.columns = ['AORC_1km - AORC_10km', 'MPE_4km - MPE_10km']
#%
comp_convert = (
    # comp_df.set_index('product', append=True)  # set product as
    # .stack()  
    comp_df.stack()                      # pull AORC, MPE into rows 
      .to_frame()                   # convert to a dataframe
      .reset_index()                # make the index into reg. columns
      # .rename(columns={'level_0': 'var', 0: 'value'})  # rename columns
      # .drop('level_2', axis='columns') 
       .rename(columns={'level_1': 'product', 0: 'value'})  # rename columns
       .drop('level_0', axis='columns')   # drop junk columcomp_df.stack()
      )

comp_convert['value']=comp_convert['value'].astype(float)
comp_convert['product']=comp_convert['product'].astype(str)

#%%
fig = plt.figure(figsize=(10,7))
fig.suptitle('Bias between products of different resolutions for heavy rains', fontsize=18)
colors = ["lightblue","lightgreen"]
sns.set_palette(sns.color_palette(colors))
ax1=sns.boxplot(x="product", y="value", data=comp_convert)
ax1 = sns.swarmplot(x="product", y="value", data=comp_convert, color=".25")
ax1.set_xlabel('')
# ax1.set_xlabel('PBias (%)', fontsize=14, weight='bold')
ax1.set_ylabel('PBias (%)', fontsize=14, weight='bold')
ax1.axhline(y=0, color='r', linestyle='--')
#ax2.yaxis.set_label_position("right")
# ax1.set_xlabel('PBias (%)', fontsize=14, weight='bold')
ax1.axhline(y=0, color='r', linestyle='--')
ax1.set_ylim([-50,450])
# ax1.set_ylabel('')
# ax1.set_xticklabels([])
ax1.tick_params(axis='both', labelsize=14)
# ax1.legend().set_visible(False)
ax1.legend(loc='lower right', ncol=2, mode="expand", borderaxespad=0).set_title('')
#ax1.set(ylabel='Metric (mm/day)')
plt.savefig(os.path.join(anal_root,'Figures\\OLand\\Diff_Extreme_Res10.jpg'),bbox_inches='tight')
plt.show()







