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
## Main information and sidebar control panel setup
st.title("Spatial clustering analysis: a tutorial")

# Page layout options
option = st.sidebar.radio(
    "Choose a page:",
    ('Introduction', 'Datasets', 'Clustering intro', 'Analysis and results', 'Conclusion')
)



########################################################################
## PAGE 1: INTRODUCTION
########################################################################
if option == 'Introduction':
    st.image('Input images/cover.jpg')
    st.header(":blue[Welcome!]")

    st.markdown('As global environmental challenges continue to escalate, understanding how to most effectively leverage modern technology innovations for conservation impact becomes increasingly critical. Each year, **WILD**LABS surveys the global conservation tech community to find out what you all are working on, what challenges you\'re facing, what support you need, and what you foresee on the horizon. Our aims in this research are to build an evidence base to share back with and support the community, to use the insights produced to create more informed and effective **WILD**LABS programs, and to communicate shared priorities to influence policy and funding decisions that will benefit our sector as a whole.  \n  \nFor the State of Conservation Technology 2023 report, we\'ve built on our 2021 results to conduct a three-year trends analysis,  bringing you insights for the first time into how dynamics have been evolving across the community over time. By highlighting shifting opinions as well as stabilizing trends in technology usage, user and developer challenges, opportunities for growth, and more, we aimed to illuminate the most useful information for advancing the sector together in a more effective and inclusive way. As always, our hope with this research is to amplify a united voice to drive progress toward impactful solutions for the planet.')
    
########################################################################
## PAGE 2: DATASETS
########################################################################
elif option == 'Datasets':
    st.image('Input images/datasets.jpg')
    st.subheader(":red[What should we look for?]")


    st.caption('*\*Data sources: [Census Bureau](https://www.census.gov/programs-surveys/acs), [Department of Energy](https://afdc.energy.gov/fuels/electricity_locations.html#/find/nearest?fuel=ELEC), [MIT Election Data and Science Lab](https://electionlab.mit.edu/data).*')

########################################################################
## PAGE 3: CLUSTERING
########################################################################    

elif option == 'Clustering intro':
    st.image('Input images/clustering.jpg')
    st.header("What is clustering?")
    
    st.divider()
    st.header("K-means method")

    st.caption('*\*Sources: [YouTube channel](https://www.youtube.com/channel/UCrxw8iiyFalKHFNAhZYCAYA/videos)*.')

########################################################################
## PAGE 4: ANALYSIS
######################################################################## 
elif option == 'Analysis and results':
    st.image('Input images/results.jpg')
    st.header("Analysis plan")
    
########################################################################
## PAGE 5: CONCLUSIONS
######################################################################## 
elif option == 'Conclusion':
    st.image('Input images/conclusion.jpg')


