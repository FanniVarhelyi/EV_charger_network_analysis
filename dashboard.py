#import and load packages
import pandas as pd
import streamlit as st
from plotnine import ggplot, aes, geom_bar, scale_y_continuous, geom_text, coord_flip, theme, element_text, labs, scale_fill_manual, theme_minimal, geom_point, geom_line, position_stack, element_rect
import matplotlib.pyplot as plt
import geopandas as gpd
import plotly.express as px
import matplotlib.patches as mpatches
from plotly.subplots import make_subplots
import seaborn as sns
import json

#import the data
@st.cache_data
def load_data(filename):
    return pd.read_csv(filename)
df = load_data('Input files/final_data.csv')
charger_snap = load_data('Input files/charger_snap.csv')
sum_stats = load_data('Input files/sum_stats.csv')
clustering = load_data('Input files/clustering_results.csv')
clust_stats = load_data('Input files/cluster_stats.csv')

@st.cache_data
def load_json_data(filename):
    with open(filename, 'r') as infile:
        data = json.load(infile)
    return data

counties = load_json_data('Input files/counties.json')

# Note: the data is already preprocessed which is not included in this file (will be included in the final project submission)


@st.cache_data
def load_geodata(filename):
    return gpd.read_file(filename)

map = load_geodata('Input files/map.gpkg')


## Data preprocessing
df['FIPS'] = df['FIPS'].astype(int).astype(str).str.zfill(5)


########################

# Page layout options
st.sidebar.title("Spatial clustering analysis: a tutorial")

option = st.sidebar.radio(
    "Pages:",
    ('Introduction', 'Datasets', 'Clustering intro', 'Analysis and results', 'Conclusion', 'References')
)
st.sidebar.title("About")
st.sidebar.info('This is a tutorial and demonstration on how to use publicly available data and unsupervised machine learning to analyze the EV network in the U.S. You can find all the corresponding code in this [GitHub](https://github.com/FanniVarhelyi/EV_charger_network_analysis.git) repo.\n\nDeveloped by Fanni Varhelyi')



########################################################################
## PAGE 1: INTRODUCTION
########################################################################
if option == 'Introduction':
    st.image('Input images/cover.jpg')
    st.subheader(":green[Welcome!]")

    st.markdown('If you\'re here, you\'re probably interested in either climate change related data science, spatial analysis of green infrastructure in the U.S., or in electric cars. The goal of this short tutorial is to showcase how analysis could be performed on publicly easily available datasets, and how such analysis can inform public policy decision-making.\n\nHopefully, this project can aid to help researchers, policymakers, or anyone with a basic programming skillset learn about how to leverage easily accessible datasets and unspervised machine learning methods to better understand current trends in the United States EV networks. This is intended only as an introduction and a starting point point when thinking about or analyzing the expansion of the EV network in the coming years.')

    st.subheader(":green[Why is the EV network important?]")
    st.markdown('Climate change is an immediate and existential threat to our society that we must address within a relatively short time period. A study from 2015 estimates that by 2050, greenhouse gas emission reduction from EV adoption could reach approximately 1.5 billion tons of CO2 per year (Lutsey, 2015). Any potential future benefit is largely dependent on how quickly the technology will be adapted, which, in turn, depends on several factors as multiple authors point out, ranging from technical specifications such as battery range or charging network density to social ones (e.g., Tran et al., 2012; Barkenbus, 2020).\n\nA key constraint to further EV adoption is the density of the charging network (mentioned by e.g., Tran et al., 2012). Expanding this network is also a priority in recent U.S. policy: in February, the White House detailed its plan on how to spend $7.5 billion on EV charging, as originally outlined in the Bipartisan Infrastructure Law (The White House, 2023).\n\nOne further factor potentially influencing this overall view  is the distribution of both the vehicles themselves, and the charging stations, and any disparities associated with either. These disparities have also been investigated by researchers: for instance, Roy and Law looked at charging station placement disparities in Orange County, California (2020).\n\nOn the following pages, I will briefly introduce the data we\'re working with, the methodology (which will be a commonly used clustering algorithm called k means), and discuss the results. I will also create some map-based visualizations both for the data and the results.')
    
########################################################################
## PAGE 2: DATASETS
########################################################################
elif option == 'Datasets':
    st.image('Input images/datasets.jpg')
    st.subheader(":green[What should we look for?]")

    st.markdown('When thinking about any analysis, it\'s important to ensure that our data is good quality, comes from a reputable source, and is up to date. While it\'s not always possible to achieve all of these, it\'s important to keep this in mind. For this project, we\'re interested in the electric charging network present in the United States, as well as some additional information about the people living at specific locations.\n\n**Dataset 1: EV charger locations**\n\nThis information can be accessed through various sources by end users, such as route planning apps. To get data that we can aanalyze and work with, I will leverage a dataset published by the U.S. Energy of Department. This dataset contains the location of each alternate fuel source in the United States, including EV chargers. After filtering everything else out, we can access the precise location, street address, provider, and type of charger among other qualities. There are 58 857 chargers in this list.\n\n**Dataset 2: Demographics of the United States**\n\nI also mentioned potentially being interested in disparaties. When looking for socio-economics data, the survey-based information periodically published by the U.S. Census Bureau is typically a good source. In this case, we will work with data sourced from the American Consumer Survey. We will leverage 5-year averaged results, and focus on key potential attributes: total population, household size, vehicle ownership, race & ethnicity and poverty on a census tract level. Since the information is on a census tract level, our second dataset contains information on 72 877 tracts. \n\n**Dataset 3: Election results on a county level**\n\nFinally, given the polarization about climate change between the political parties in the U.S., I thought it might be an interesting additional aspect to look at party affiliation. To do so, I will leverage a Presidential elections dataset published by MIT. This dataset includes state, county, year, and the number of votes cast for each candidate since 2000. I will focus on the latest election, and will categorize counties and states as either Republican and Democratic based on the winner of the latest Presidential election in that given county or state.')

    st.divider()
    st.subheader(":green[Snapshot: EV charger data]")

    st.markdown('The primary data source for this research is the location of electric vehicle charging stations in the United States, which is available from the U.S. Department of Energy. This dataset contains the location of 58 857 charging stations and contains 71 further attributes. Of these attributes, the most important ones for this project will be location. Here\'s a summary look on the state level of the available total charging stations:')

    st.table(charger_snap.iloc[:9,:])

    

    

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

    st.divider()
    st.subheader(":green[Combining all of our data for analysis]")

    st.markdown('When working with spatial data, it\'s important to understand the different levels of aggregation (geometries). For example, when thinking about the United States and working with census data, we can look at states, counties, census tracts, census blocks. Some data is available on the census tract level which is usually used for analysis. In our case, we have the zip code and exact coordinates of each charging station in the U.S., and we can check which census tract these points correspond to. To do so, we need to work with shapefiles. Shapefiles are, in essence, datasets that contain a specific column that outlines the border of a polygon shape (for example, coordinates of the border of a state or country). When we have a shapefile like this, we can check if any point is included within its boudaries, and count how many points we found. We will need the gopandas package to achieve this.\n\nFor our project, we will leverage the shape files associated with census tracts (also published by the Census Bureau), and for each EV charger station, we will match the point with the polygon shape and its I.D. that contains it. Once we concluded this, we can change the unit of our analysis from individual points to census tracts by aggregating all our information on the census tract level, using the acquired I.D.s, called GEOID that corresponds to each census tract.')

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

    st.markdown('Once we have the number of chargers per census tract, we can easily merge this with any census variables we would like to include in our analysis. In this case, I selected total population, cars per household, poverty, and racial & ethnic attributes. When working with data like this, it\'s often useful to check the summary statistics for the relevant variables:')

    st.table(sum_stats.iloc[:4,:5])
    
    st.markdown('Unfortunately, when looking at our third dataset, we can see that information is not available on a census tract level. This is understadable: tracts are smaller than voter districts. Thus, we will need to aggregate again to have the same unit of analysis for all of our data. In this case, this will be a county level. Once we\'ve done this, and selected the relevant variables, we have our data ready for analysis!')

    st.table(df.head(5))

    st.markdown('Another interesting way to look at our data and understand it\'s attributes is using visualizations. The next graph showcases the distribution of a given variable, while the following map of the United States shows the county-level value of a selected variable.')

    ####ADD BOXPLOT
    variables = [
       '% of households with 2 or more cars',
       'Number of Level 3 chargers', 'Number of Level 1 chargers',
       'Number of Level 2 chargers',
       '% of Non-Hispanic Black population',
       '% of Hispanic population',
       '% of the population in poverty',
       '% of white population']
    choice = st.selectbox('Variable', variables)
    
    fig2 = plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[choice])
    plt.title('Exploratory data analysis')
    plt.xlabel(f'Variable: {choice}')
    st.pyplot(fig2)


    #####ADD MAP
    variables = ['Political party (county)',
       '% of households with 2 or more cars',
       'Number of Level 3 chargers', 'Number of Level 1 chargers',
       'Number of Level 2 chargers',
       '% of Non-Hispanic Black population',
       '% of Hispanic population',
       '% of the population in poverty',
       '% of white population']
    choice2 = st.selectbox('Variable ', variables)

    if choice2 == 'Political party (county)':
    # Define the discrete color map for political parties
        color_map = {'REPUBLICAN': 'red', 'DEMOCRAT': 'blue'}

        fig = px.choropleth(df, geojson=counties, locations='FIPS', color=choice2,
                            color_discrete_map=color_map,
                            scope="usa",
                            labels={choice2: 'Political Party'})
    else:
        # Use a continuous color scale
        fig = px.choropleth(df, geojson=counties, locations='FIPS', color=choice2,
                            color_continuous_scale="Viridis",
                            scope="usa",
                            labels={choice2: choice2})

    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    #fig.show()
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('Looking at counties on the country level, however, can make it difficult to understand local characteristics, so we can also zoom in on a given state and then look at our variables. Exploratory data analysis and visualizations such as these can help us better understand the data, which in turn will influence some of our design choices later on for the analysis.')

    ##ADD MAP ON STATE LEVEL
    choice3 = st.selectbox('Variable:', variables)
    choice4 = st.selectbox('State', df.State.unique().tolist())

    filtered = df[df['State'] == choice4]
    if choice3 == 'Political party (county)':
        color_map = {'REPUBLICAN': 'red', 'DEMOCRAT': 'blue'}
        fig = px.choropleth(filtered, geojson=counties, locations='FIPS', color=choice3,
                                color_discrete_map=color_map,
                                scope="usa",
                                labels={choice3}
                                )
        fig.update_geos(fitbounds="locations", visible=False)
    else:
        fig = px.choropleth(filtered, geojson=counties, locations='FIPS', color=choice3,
                                color_continuous_scale="Viridis",
                                scope="usa",
                                labels={choice3}
                                )
        fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)


    st.caption('*Data sources: [Census Bureau](https://www.census.gov/programs-surveys/acs), [Department of Energy](https://afdc.energy.gov/fuels/electricity_locations.html#/find/nearest?fuel=ELEC), [MIT Election Data and Science Lab](https://electionlab.mit.edu/data).*')

########################################################################
## PAGE 3: CLUSTERING
########################################################################    

elif option == 'Clustering intro':
    st.image('Input images/clustering.jpg')

    st.subheader(":green[A few words on unsupervised machine learning]")

    st.markdown('Machine learning is a useful tool we can deploy to analyze any kind of data. It\'s relatively straightforward when we have a target, outcome, or class label we\'re interested in. For example, we could use a model to determine if a tweet is positive or negative based on the words in contains. We can also leverage machine learning if we\'d like to learn about a set of attributes and how they relate to each other without having a specific end goal in mind. We could simplify dozens or hundreds of attributes to a few key summary metrics, or group observations into clusters based on their characteristics.')

    st.divider()
    st.subheader(":green[What is clustering?]")

    st.markdown('Clustering is a well-known unsupervised machine learning method used to identify homogeneous groups using information within a dataset.\n\nClustering can help us identify if there are any subsets of the data that belong together based on some shared attributes or set of attributes. Even if we don\'t know how many clusters there are, algorithms can help divide our data to as many groups as many makes sense. These groupring are typically made by trying to minimize within-cluster variation, or SSE, or potentially by maximizing variation across clusters. The below illustration shows how we can find different number of clusters in 2-dimensional data. When we only have 2 or 3 dimension (2 or 3 variables) we can even look at the data and determine clusters by ourselves. Once we\'re working in higher dimensions, this becomes impossible. As our data has 10 variables we will use for clustering (more on this later), we can\'t identify clusters thies easily anymore.')

    st.image('Input images/clusters.jpg')
    
    st.divider()
    st.subheader(":green[K-means method]")

    st.markdown('K-means separates data into a pre-specified number of clusters. It assigns each observation to one cluster (so no observation can belong to multiple clusters) and it minimizes within-cluster variation for non-overlapping clusters. A key characteristic of this method is that we need to pick the number of clusters (k) upfront. We will see in the analysis section how we can determine the optimal number of k with our dataset.')

    st.caption('*\*Sources: Data Science III course, 2023 Fall semester, Georgetown University*.')

########################################################################
## PAGE 4: ANALYSIS
######################################################################## 
elif option == 'Analysis and results':
    st.image('Input images/results.jpg')
    st.subheader(":green[Analysis]")

    st.markdown('For our analysis, we will leverage the popular machine learning library scicit learn, or sklearn. Sklearn can help both preprocess our data and implement the clustering model.\n\nPreprocessing is important when clustering: it ensures we account for differences in scale. In our case, many of our variables are already on the same scale: we`re using percentages for motiple variables. Regardless, it makes sense to standardize all variables.\n\nBefore this step, we also need to drop all columns that contain information that is unnecessary for the clustering itself. These are usually columns containing identifying information, such as the county name or county fips number. I also decided to not inlcude state as a variable. Thus, the final variable list is as follows:')

    #check if works!!!
    st.markdown('- Vehicle ownership: percentage of households with 2 or more cars\n- Number of Level 3 chargers\n- Number of Level 2 chargers\n- Number of Level 1 chargers\n- Percentage of white population\n- Percentage of Non-Hispanic Black population\n- Percentage of Hispanic population\n- Percentage of the population below the poverty line\n- County political affiliation\n- state political affiliation')
                
    st.markdown('Once we have selected only these variables, we can then normalize our data:')

    code3 = '''
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    x = ev_chargers.drop(columns=['county_fips', 'state_y',
                              'county_name', 'state_po',
                              'state_party'])
    x['party'] =  x['party'].replace({'REPUBLICAN': 0, 'DEMOCRAT': 1})

    x = pd.DataFrame(StandardScaler().fit_transform(x))
    '''

    st.code(code3, language = 'python')

    st.markdown('After preprocessing the data, we can finally decide what the optimal number of clusters would be. To do so, we would like to find a number of clusters that minimizes the overall error in our clusters. In other words, we\'re looking for clusters that are well-formed, don\'t overlap, and make sense. Let\'s take a look at this overall error (the sum of squared errors or SSE to be precise) for a k between 1 and 16.')

    st.image('Input images/sse.jpg')

    st.markdown('An optimal number would probably be 7 clusters. Using this information, we can run the clustering algorithm with k = 7.')

    code4 = '''
    km = KMeans(n_clusters=7, init='random', random_state=0)
    km_clusters = km.fit_predict(x)
    '''

    st.code(code4, language = 'python')

    st.markdown('We can see a snapshot of the results below.')

    st.table(clustering.sample(10))

    st.subheader(":green[Results]")

    st.markdown('To understand the results, we could, for example, look at the summary statistics again, but by clusters.')

    st.table(clust_stats[:-1])

    st.markdown('Alternatively, we could also take a look at a map again to visualize the clusters.')

    variables = ['Cluster',
                 'Political party (county)',
       '% of households with 2 or more cars',
       'Number of Level 3 chargers', 'Number of Level 1 chargers',
       'Number of Level 2 chargers',
       '% of Non-Hispanic Black population',
       '% of Hispanic population',
       '% of the population in poverty',
       '% of white population']
    choice5 = st.selectbox('Variable or cluster:', variables)

    if choice5 == 'Political party (county)':
    # Define the discrete color map for political parties
        color_map = {'REPUBLICAN': 'red', 'DEMOCRAT': 'blue'}

        fig = px.choropleth(df, geojson=counties, locations='FIPS', color=choice5,
                            color_discrete_map=color_map,
                            scope="usa",
                            labels={choice5: 'Political Party'})
    elif choice5 == 'Cluster':
        df['Cluster'] = df['Cluster'].astype(str)
        # Define a discrete color map for clusters
        # You can customize this list of colors as needed
        cluster_colors = px.colors.qualitative.Plotly

        fig = px.choropleth(df, geojson=counties, locations='FIPS', color=choice5,
                        color_discrete_sequence=cluster_colors[:7],
                        scope="usa",
                        labels={choice5: 'Clusters'})
    else:
        # Use a continuous color scale
        fig = px.choropleth(df, geojson=counties, locations='FIPS', color=choice5,
                            color_continuous_scale="Viridis",
                            scope="usa",
                            labels={choice5: choice5})

    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    #fig.show()
    st.plotly_chart(fig, use_container_width=True)
    
########################################################################
## PAGE 5: CONCLUSIONS
######################################################################## 
elif option == 'Conclusion':
    st.image('Input images/conclusion.jpg')

    st.markdown('In this tutorial, we learned how to merge publicly available datasets with different levels of aggregation, and how to use clustering to start to understand some patterns in EV charger placements in the United States.\n\nOur results indicate California counties are outliers even compared to other rich, Democrat-leaning counties. This can be alarming given that California is just a small subset of the country and EV adoption in the rest of the country could have lead to, for example, big cities in other states being similar to big cities in California.\n\nAnother key finding is that county political affiliation matters: most counties clustered together (for clusters 1, 2, 3, 5, 7) have very similar or the same political affiliations.\n\nThese preliminary results can help guide further research. For example, we could investigate if there are other characteristics neglegted here that could be important. We could look at the different cluster attributes and think how these could influence policymaking. For example, it seems like Republican counties tend to be similar and might even have less charging stations. Federal policy could focus on incentives to increase adoption in these areas. We could also look at the results on poverty and race & ethnicity, and further try to understand where EV chargers lag behind (such as in cluster 4, with smaller number of charging stations and high poverty). We could zoom in on these counties and try to better understand what the issue is, and if state-level incentives could help increase the number of charging stations. We should probably also look at population density, overall population and roads in any given county to understand them better.')

########################################################################
## PAGE 6: REFS
######################################################################## 
elif option == 'References':
    st.markdown('Bibliography:\n\nBarkenbus, J. N. (2020). Prospects for Electric Vehicles. Sustainability 12, no. 14: 5813.\n\nEl Rai, M.C., Hadi, S.A., Damis, H. A. and Gawanmeh, A. (2022). Prediction of Electric Vehicle Charging Stations Distribution Using Machine Learning. 2022 5th International Conference on Signal Processing and Information Security (ICSPIS), Dubai, United Arab Emirates, pp. 154-157\n\nLutsey, N. (2015). Global climate change mitigation potential from a transition to electric vehicles. International Council on Clean Transportation.\n\nThe White House. (2023). FACT SHEET: Biden-⁠Harris Administration Announces New Standards and Major Progress for a Made-in-America National Network of Electric Vehicle Chargers. The White House.\n\nTran, M., Banister, D., Bishop, J. et al. (2012) Realizing the electric-vehicle revolution. Nature Clim Change 2, 328–333')
    st.markdown('Data sources: [Census Bureau](https://www.census.gov/programs-surveys/acs), [Department of Energy](https://afdc.energy.gov/fuels/electricity_locations.html#/find/nearest?fuel=ELEC), [MIT Election Data and Science Lab](https://electionlab.mit.edu/data).')


