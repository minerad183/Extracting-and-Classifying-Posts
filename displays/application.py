import streamlit as st
import pandas as pd
import geopandas as gpd
import pickle
import pyproj
import plotly.graph_objs as go
import json
import shapely
import streamlit.components.v1 as components

#Read in data
clusters_df = pd.read_csv('./data/get_geoconfirmed_data_clusters.csv', encoding="utf-8")
clusters_df['geometry'] = clusters_df.apply(lambda row: shapely.Point(row['longitude'], row['latitude']), axis=1)
clusters_df = gpd.GeoDataFrame(clusters_df, geometry='geometry')
clusters_df = clusters_df.set_crs("EPSG:4326", allow_override=True)
# define lat, long for points
Lat = clusters_df['latitude']
Long = clusters_df['longitude']

# set GeoJSON file path
path = './data/geojson.json'
# write GeoJSON to file
clusters_df.to_file(path, driver = "GeoJSON", encoding='utf-8')
with open(path, encoding = 'utf-8') as geofile:
   j_file = json.load(geofile)
# index geojson
i=1
for feature in j_file["features"]:
   feature ['id'] = str(i).zfill(2)
   i += 1

#I used these sites to help make the interactive map: https://towardsdatascience.com/how-to-create-interactive-map-plots-with-plotly-7b57e889239a
# https://towardsdatascience.com/build-a-multi-layer-map-using-streamlit-2b4d44eb28f3
# mapbox token
mapboxtoken = 'pk.eyJ1IjoibWluZXJhZCIsImEiOiJjbGhnYm5nZGEwM2JjM3FwbjBnbnN4cHQ4In0.r_9syT8uhNvtxeH4unVpRg'

data = []

for event in clusters_df['clusters'].unique():
    if event == 0:
        event_data = dict(
        lat = clusters_df.loc[clusters_df['clusters'] == event,'latitude'],
        lon = clusters_df.loc[clusters_df['clusters'] == event,'longitude'],
        name = 'Russian Movements and Activities',
        marker = dict(size = 8, opacity = 0.5),
        type = 'scattermapbox'
        )
    elif event == 1:
        event_data = dict(
        lat = clusters_df.loc[clusters_df['clusters'] == event,'latitude'],
        lon = clusters_df.loc[clusters_df['clusters'] == event,'longitude'],
        name = 'Global Russian and Ukrainian Activities',
        marker = dict(size = 8, opacity = 0.5),
        type = 'scattermapbox'
        )
    elif event == 2:
        event_data = dict(
        lat = clusters_df.loc[clusters_df['clusters'] == event,'latitude'],
        lon = clusters_df.loc[clusters_df['clusters'] == event,'longitude'],
        name = 'The Siege of Mariupol',
        marker = dict(size = 8, opacity = 0.5),
        type = 'scattermapbox'
        )
    elif event == 3:
        event_data = dict(
        lat = clusters_df.loc[clusters_df['clusters'] == event,'latitude'],
        lon = clusters_df.loc[clusters_df['clusters'] == event,'longitude'],
        name = 'The Destruction Cluster',
        marker = dict(size = 8, opacity = 0.5),
        type = 'scattermapbox'
        )
    elif event == 4:
        event_data = dict(
        lat = clusters_df.loc[clusters_df['clusters'] == event,'latitude'],
        lon = clusters_df.loc[clusters_df['clusters'] == event,'longitude'],
        name = 'Ukrainian Positions and Activities',
        marker = dict(size = 8, opacity = 0.5),
        type = 'scattermapbox'
        )
    elif event == 5:
        event_data = dict(
        lat = clusters_df.loc[clusters_df['clusters'] == event,'latitude'],
        lon = clusters_df.loc[clusters_df['clusters'] == event,'longitude'],
        name = 'Battle for Bakhmut',
        marker = dict(size = 8, opacity = 0.5),
        type = 'scattermapbox'
        )
    elif event == 6:
        event_data = dict(
        lat = clusters_df.loc[clusters_df['clusters'] == event,'latitude'],
        lon = clusters_df.loc[clusters_df['clusters'] == event,'longitude'],
        name = 'Satellite Imagery',
        marker = dict(size = 8, opacity = 0.5),
        type = 'scattermapbox'
        )                
    data.append(event_data)

layout = dict(title_text ='Russia-Ukraine Conflict Interactive Map', title_x =0.5,  
         width=950, height=700,mapbox = dict(center= dict(lat=47,  
         lon=35),accesstoken= mapboxtoken, zoom=4,style="carto-positron"))

clusters_names = clusters_df.clusters

st.title("Social Media Posts of the Russia-Ukraine Conflict")
st.subheader("Clustering Analysis")

page = st.sidebar.selectbox(
    'Pages',
    ('About', 'EDA', 'Predicting', 'Interactive Map')
)

if page == 'About':
    st.subheader('''Extracting, Modeling, Classifying, and Displaying Geolocated Social Media Posts from the Russia-Ukraine War''')
    st.write('''Russia's February 2022 invasion of Ukraine marked the end of a two-decade peace in Europe, and is the largest land war in Europe since World War II. This war is also one of the first instances of a war fought in the social media and information space, as well. With the number of smart phones and people connected to the internet, both in Ukraine and around the world, the Russia-Ukraine conflict has been cataloged like no other war before it. Now media organizations are not the only ones covering the war, everyday people can do it by just taking a picture or a video and posting it to social media sites like Twitter, or Telegram. Whole ecosystems have sprouted up to facilitate it, such as individual users on Twitter aggregating media posts, to loosely moderated threads on message boards like Reddit.''')
    st.write('''There are also non-government organizations trawling through social media and capturing as much information about the posts. Analysts employed by these organizations use their skills and expertise to construct narratives and fill in the gaps that the data from social media might have missed. In many cases, these organizations are also making their data freely available to other organizations and governments for their own use. Some examples of these organizations are Bellingcat, Texty.ua, C4ADS, and @GeoConfirmed. These disparate organizations are using similar sources to analyze and display their data, but data may be missed by one or more. That is why it is imperative to utilize all that is out there to obtain a complete picture to track the conflict.''')
    st.write('''Social media data represents a necessary component of Ukraine battle tracking. With all of these organizations doing the hard part of tracking and gathering the data, it falls on other groups to analyze the data in a meaningful way. There currently exist a few problems that this capstone project hopes to rectify:''')
    expander  = st.expander('Expand')
    expander.markdown("- The data are not made available in a one-stop-shop location, and require data science expertise to gather, extract-transfer-load, and store.")
    expander.markdown("- The way the data are collected and labeled is not standardized across different sources, and need to be standardized in a more digestable format.")
    expander.markdown("- The data and its contents can be analyzed more holistically, across space and time, and models built out of it.")
    expander.markdown("- The data should be displayed in an understandable, relevant, and useful format, such as through charts and maps, on a, interactive website dedicated to this.")

elif page == 'Interactive Map':
    updatemenus=list([
        # drop-down 1: map styles menu
        dict(
            buttons=list([
                dict(
                    args=['mapbox.style', 'dark'],
                    label='Dark',
                    method='relayout'
                ),                    
                dict(
                    args=['mapbox.style', 'light'],
                    label='Light',
                    method='relayout'
                ),
                dict(
                    args=['mapbox.style', 'outdoors'],
                    label='Outdoors',
                    method='relayout'
                ),
                dict(
                    args=['mapbox.style', 'satellite-streets'],
                    label='Satellite with Streets',
                    method='relayout'
                )                    
            ]),
            # direction where I want the menu to expand when I click on it
            direction = 'up',
        
            # here I specify where I want to place this drop-down on the map
            x = 0.75,
            xanchor = 'left',
            y = 0.05,
            yanchor = 'bottom',
        
            # specify font size and colors
            bgcolor = '#000000',
            bordercolor = '#FFFFFF',
            font = dict(size=11)
        ),    
        
        # drop-down 2: select type of event event to visualize
        dict(
            buttons=list([
                dict(label = 'All Clusters',
                    method = 'update',
                    args = [{'visible': [True, True, True, True, True, True, True]}]),
                dict(label = 'Russian Movements and Activities',
                    method = 'update',
                    args = [{'visible': [False, False, False, False, False, False, True]}]),
                dict(label = 'Global Russian and Ukrainian Activities.',
                    method = 'update',
                    args = [{'visible': [True, False, False, False, False, False, False]}]),
                dict(label = 'The Siege of Mariupol',
                    method = 'update',
                    args = [{'visible': [False, False, False, False, False, True, False]}]),
                dict(label = 'The Destruction Cluster',
                    method = 'update',
                    args = [{'visible': [False, False, False, True, False, False, False]}]),
                dict(label = 'Ukrainian Positions and Activities',
                    method = 'update',
                    args = [{'visible': [False, True, False, False, False, False, False]}]),
                dict(label = 'Battle for Bakhmut',
                    method = 'update',
                    args = [{'visible': [False, False, True, False, False, False, False]}]),
                dict(label = 'Satellite Imagery',
                    method = 'update',
                    args = [{'visible': [False, False, False, False, True, False, False]}])
            
            ]),
            # direction where the drop-down expands when opened
            direction = 'down',
            # positional arguments
            x = 0.01,
            xanchor = 'left',
            y = 0.99,
            yanchor = 'bottom',
            # fonts and border
            bgcolor = '#000000',
            bordercolor = '#FFFFFF',
            font = dict(size=11)
        )
    ])

    # assign the list of dictionaries to the layout dictionary
    fig = dict(data = data, layout=layout)
    layout['updatemenus'] = updatemenus

    st.plotly_chart(fig, use_container_width = True)