# Precip.-Analysis
Python Scripts for Precipitation Analysis

This folder contains scripts for the paper Hong et al. 2022 JOH titled:
"Evaluation of gridded precipitation datasets over international basins and large lakes"
https://www.sciencedirect.com/science/article/pii/S0022169422000828

Code structure:
1, Data pre-treatment: analyze the raw datasets of AORC, MPE, CaPA, Merged CaPA-MPE, and gauge_observations, and preparation for data analysis:
AORC.py, CaPA_mrcc.py, MPE_mrcc.py, Merged_mrcc.py, OL_gauges.py

2, Data analysis:
- Analyse_OLand2.py: Overland Precipitation analysis
- Analyse_Olake.py: Overlake Precipitation analysis
- Analyse_extreme.py: Extreme precipitation analysis
- Analyse_Country.py: Precipitation Analysis by Country
- Analyse_AHHCD.py: Analysis of the AHHCD datasets

3, precip_functions.py: Contains functions used in the above scripts
