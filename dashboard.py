#import and load packages
import pandas as pd
import streamlit as st
from plotnine import ggplot, aes, geom_bar, scale_y_continuous, geom_text, coord_flip, theme, element_text, labs, scale_fill_manual, theme_minimal, geom_point, geom_line, position_stack, element_rect
import matplotlib.pyplot as plt
import geopandas as gpd
import plotly.express as px
import matplotlib.patches as mpatches
from plotly.subplots import make_subplots

#import the data
@st.cache_data
def load_data(filename):
    return pd.read_csv(filename)
df = load_data('Input files/final_data.csv')

# Note: the data is already preprocessed which is not included in this file (will be included in the final project submission)


@st.cache_data
def load_geodata(filename):
    return gpd.read_file(filename)

map = load_geodata('Input files/map.gpkg')


########################

# Page layout options
option = st.sidebar.radio(
    "Spatial clustering analysis: a tutorial\n\nTable of contents:",
    ('Introduction', 'Datasets', 'Clustering intro', 'Analysis and results', 'Conclusion', 'References')
)



########################################################################
## PAGE 1: INTRODUCTION
########################################################################
if option == 'Introduction':
    st.image('Input images/cover.jpg')
    st.subheader(":green[Welcome!]")

    st.markdown('Climate change is an immediate and existential threat to our society that we must address within a relatively short time period. A study from 2015 estimates that by 2050, greenhouse gas emission reduction from EV adoption could reach approximately 1.5 billion tons of CO2 per year (Lutsey, 2015). Any potential future benefit is largely dependent on how quickly the technology will be adapted, which, in turn, depends on several factors as multiple authors point out, ranging from technical specifications such as battery range or charging network density to social ones (e.g., Tran et al., 2012; Barkenbus, 2020).\n\nA key constraint to further EV adoption is the density of the charging network (mentioned by e.g., Tran et al., 2012). Expanding this network is also a priority in recent U.S. policy: in February, the White House detailed its plan on how to spend $7.5 billion on EV charging, as originally outlined in the Bipartisan Infrastructure Law (The White House, 2023).\n\nOne further factor potentially influencing this overall view  is the distribution of both the vehicles themselves, and the charging stations, and any disparities associated with either. These disparities have also been investigated by researchers: for instance, Roy and Law looked at charging station placement disparities in Orange County, California (2020).\n\nThe goal of this project is to help researchers, policymakers, or anyone with a basic programming skillset learn about how to leverage easily accessible datasets and unspervised machine learning methods to better understand current trends in the United States EV networks. Hopefully, this tutorial provides a useful starting point when thinking about or analyzing the expansion of the EV network in the coming years.')
    
########################################################################
## PAGE 2: DATASETS
########################################################################
elif option == 'Datasets':
    st.image('Input images/datasets.jpg')
    st.subheader(":green[What should we look for?]")

    st.markdown('When thinking about any analysis, it\'s important to ensure that our data is good quality, comes from a reputable source, and is up to date. While it\'s not always possible to achieve all of these, it\'s important to keep this in mind.\n\nIn this case, we\'re interested in the electric charging network present in the United States. This infromation can be accessed through various sources by end users, such as route planning apps. To get data that we can aanalyze and work with, I will leverage a dataset published by the U.S. Energy of Department.\n\nI also mentioned potentially being interested in disparaties. When looking for socio-economics data, the survey-based information periodically published by the U.S. Census Bureau is typically a good source. In this case, we will work with data sourced from the American Consumer Survey.\n\nFinally, given the polarization about climate change between the political parties in the U.S., I thought it might be an interesting additional aspect to look at party affiliation. To do so, I will leverage a dataset upblished by MIT that includes the votes cast in the last Presidential election. I will categorize counties and states based on this information.')

    st.subheader(":green[Snapshot: EV charger data]")

    st.markdown('The primary data source for this research is the location of electric vehicle charging stations in the United States, which is available from the U.S. Department of Energy. This dataset contains the location of 58 857 charging stations and contains 71 further attributes. Of these attributes, the most important ones for this project will be location. Here\'s a summary look on the state level of the available total charging stations:')

    latex_table = r'''
        \begin{tabular}{|l|r|r|r|r|r|r|r|r|r|}
        \hline
        State & Total & ChargePoint Network & Tesla Destination & Non-Networked & Blink Network & FLO & Tesla & eVgo Network & Other \\ \hline
        CA    & 15312 & 10385              & 734               & 849           & 681           & 459 & 396   & 358          & 1450  \\ \hline
        NY    & 3632  & 1885               & 471               & 268           & 160           & 58  & 83    & 25           & 682   \\ \hline
        FL    & 3248  & 1363               & 341               & 229           & 599           & 1   & 145   & 27           & 543   \\ \hline
        TX    & 2943  & 1422               & 298               & 171           & 383           & 0   & 137   & 64           & 468   \\ \hline
        MA    & 2673  & 2177               & 51                & 99            & 76            & 25  & 46    & 19           & 180   \\ \hline
        CO    & 2044  & 1315               & 94                & 165           & 228           & 3   & 37    & 32           & 170   \\ \hline
        WA    & 2009  & 1003               & 113               & 173           & 324           & 26  & 50    & 30           & 290   \\ \hline
        GA    & 1776  & 1012               & 151               & 134           & 202           & 2   & 45    & 32           & 198   \\ \hline
        PA    & 1578  & 822                & 98                & 195           & 153           & 10  & 72    & 24           & 204   \\ \hline
        MD    & 1558  & 537                & 53                & 185           & 336           & 0   & 52    & 29           & 366   \\ \hline
        \end{tabular}
        '''

    # Display the LaTeX table in Streamlit
    st.markdown(latex_table, unsafe_allow_html=True)

    st.markdown('We can see that certain networks have much greater amount of stations that others. This doesn\'t feel right knowing the basic attributes of existing providers. Thus, it might make sense to investigate the addresses: potentially, some of our data might be on the charger level, not the charging table level. This seems to be the case, and aggregation based on addresses solves the issue:')

    code = '''chargers_clean = chargers.groupby(['Street Address', 'City', 'EV Network']).agg({
        'Latitude': 'first',
        'Longitude': 'first',
        'EV DC Fast Count': 'sum',
        'EV Level1 EVSE Num' : 'sum',
        'EV Level2 EVSE Num' : 'sum',
        'Station Name' : 'first',
        'State' : 'first',
        'ZIP' : 'first'
    }).reset_index()'''

    st.code(code,language='python')

    st.subheader(":green[Combining all of our data for analysis]")

    st.markdown('When working with spatial data, it\'s important to understand the different levels of aggregation (geometries, shapes). For example, when thinking about the United States and working with census data, we can look at states, counties, census tracts, census blocks. Some data is available on the census tract level which is usually used for analysis. In our case, we have the zip code and exact coordinates of each charging station in the U.S., and we can check which census tract these points correspond to. To do so, we need to work with shapefiles. Shapefiles are, in essence, datasets that contain a specific column that outlines the border of a polygon shape (for example, coordinates of the border of a state or country). When we have a spahefile like this, we can check if any point is included within its boudaries, and count how many points we found. We will need the gopandas package to achieve this.')

    code2 = '''import geopandas as gpd
    check_charger_in_tract = gpd.sjoin(charger_list, us_census_tracts, how="inner", op='intersects')
    per_tract = per_tract.groupby('GEOID').agg({
        'EV DC Fast Count': 'sum',
        'EV Level1 EVSE Num' : 'sum',
        'EV Level2 EVSE Num' : 'sum',
        'Station Name' : 'first',
        'City' : 'first',
        'Street Address' : 'first',
        'State' : 'first',
        'ZIP' : 'first',
        'EV Network': lambda x: ', '.join(x.dropna().unique())
    }).reset_index()
    per_tract['ev_chargers'] = per_tract['ev_chargers']'''

    st.code(code2,language='python')

    st.markdown('Once we have the number of chargers per census tract, we can easily merge this with any census variables we would like to include in our analysis. In this case, I selected total population, cars per household, poverty, and racial attributes. Unfortunately, when looking at our third dataset, we can see that information is not available on a census tract level. This is understadable: tracts are smaller than voter districts. Thus, we will need to aggregate again to have the same unit of analysis for all of our data. In this case, this will be a county level. Once we\'ve done this, and selected the relevant variables, we have our data ready for analysis!')


    st.caption('*Data sources: [Census Bureau](https://www.census.gov/programs-surveys/acs), [Department of Energy](https://afdc.energy.gov/fuels/electricity_locations.html#/find/nearest?fuel=ELEC), [MIT Election Data and Science Lab](https://electionlab.mit.edu/data).*')

########################################################################
## PAGE 3: CLUSTERING
########################################################################    

elif option == 'Clustering intro':
    st.image('Input images/clustering.jpg')
    st.subheader(":green[What is clustering?]")
    
    st.divider()
    st.subheader(":green[K-means method]")

    st.caption('*\*Sources: [YouTube channel](https://www.youtube.com/channel/UCrxw8iiyFalKHFNAhZYCAYA/videos)*.')

########################################################################
## PAGE 4: ANALYSIS
######################################################################## 
elif option == 'Analysis and results':
    st.image('Input images/results.jpg')
    st.subheader(":green[Analysis plan]")
    
########################################################################
## PAGE 5: CONCLUSIONS
######################################################################## 
elif option == 'Conclusion':
    st.image('Input images/conclusion.jpg')

########################################################################
## PAGE 6: REFS
######################################################################## 
elif option == 'References':
    st.markdown('Bibliography:\n\nBarkenbus, J. N. (2020). Prospects for Electric Vehicles. Sustainability 12, no. 14: 5813.\n\nEl Rai, M.C., Hadi, S.A., Damis, H. A. and Gawanmeh, A. (2022). Prediction of Electric Vehicle Charging Stations Distribution Using Machine Learning. 2022 5th International Conference on Signal Processing and Information Security (ICSPIS), Dubai, United Arab Emirates, pp. 154-157\n\nLutsey, N. (2015). Global climate change mitigation potential from a transition to electric vehicles. International Council on Clean Transportation.\n\nThe White House. (2023). FACT SHEET: Biden-⁠Harris Administration Announces New Standards and Major Progress for a Made-in-America National Network of Electric Vehicle Chargers. The White House.\n\nTran, M., Banister, D., Bishop, J. et al. (2012) Realizing the electric-vehicle revolution. Nature Clim Change 2, 328–333')
    st.markdown('Data sources: [Census Bureau](https://www.census.gov/programs-surveys/acs), [Department of Energy](https://afdc.energy.gov/fuels/electricity_locations.html#/find/nearest?fuel=ELEC), [MIT Election Data and Science Lab](https://electionlab.mit.edu/data).')


