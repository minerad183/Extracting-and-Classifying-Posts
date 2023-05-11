import streamlit as st
import pandas as pd
import geopandas as gpd
import pickle
import plotly.graph_objs as go
import json
import shapely
import streamlit.components.v1 as components
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import datetime as dt


st.set_option('deprecation.showPyplotGlobalUse', False)


#Read in data
clusters_df = pd.read_csv('./data/get_geoconfirmed_data_clusters.csv', encoding="utf-8")
clusters_df['geometry'] = clusters_df.apply(lambda row: shapely.Point(row['longitude'], row['latitude']), axis=1)
clusters_df = gpd.GeoDataFrame(clusters_df, geometry='geometry')
clusters_df = clusters_df.set_crs("EPSG:4326", allow_override=True)
activity_counts_df = pd.read_csv('./data/activity_counts.csv', encoding="utf-8")
activity_counts_df.reset_index(inplace = True)
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

# Stopword removal
stpwrd = stopwords.words('english')
new_stopwords = ["twitter", 'geoconfirmed', 'com', 'br', 'https', 'geo', 'png',
                 'status', 'vid', 'f', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 
                'sep', 'oct', 'nov', 'dec', 'Æ', 'ô', 'ö', 'ò', 'û', 'ù', 'ÿ', 'á', 'í', 'ó', 'ú', 'ñ', 'Ñ', 'Š', 'š', 'ý', 'ü',
                'õ', 'ð', 'ã', 'Ý', 'Ü', 'Û', 'Ú', 'Ù', 'Ï', 'Î', 'Í', 'Ì', 'Ë', 'Ê', 'É', 'È', 'Å', 'Ä', 'Ã', 'Â', 'Á', 'À', 'Ö', 'Õ', 'Ô','Ó', 'Ò',
                'ÂƒÆ', 'â', 'Âƒâ', 'šâ', 'šÂ', 'Ž', 'žÂ', 'ÃƒÆ', 'Ãƒâ', 'ƒ', 'šÃ'  ] #add update to this
stpwrd.extend(new_stopwords)

#Define a word cleaning function
def review_to_wordlist(review, remove_stopwords=True):
    # Clean the text, with the option to remove stopwords.
    
    # Convert words to lower case and split them
    words = review.lower().split()

    # Optionally remove stop words (true by default)
    if remove_stopwords:
        words = [w for w in words if not w in stpwrd]
    
    review_text = " ".join(words)

    # Clean the text
    review_text = re.sub(r"[^A-Za-z0-9(),!.?\'\`]", " ", review_text)
    review_text = re.sub(r"\'s", " 's ", review_text)
    review_text = re.sub(r"\'ve", " 've ", review_text)
    review_text = re.sub(r"n\'t", " 't ", review_text)
    review_text = re.sub(r"\'re", " 're ", review_text)
    review_text = re.sub(r"\'d", " 'd ", review_text)
    review_text = re.sub(r"\'ll", " 'll ", review_text)
    review_text = re.sub(r",", " ", review_text)
    review_text = re.sub(r"\.", " ", review_text)
    review_text = re.sub(r"!", " ", review_text)
    review_text = re.sub(r"\(", " ( ", review_text)
    review_text = re.sub(r"\)", " ) ", review_text)
    review_text = re.sub(r"\?", " ", review_text)
    review_text = re.sub(r"\s{2,}", " ", review_text)
    review_text = re.sub(r"\s{2,}", " ", review_text)
    for i in new_stopwords:
        review_text = re.sub(i, " ", review_text)
    words = review_text.split()
    
    # Shorten words to their stems
    stemmer = SnowballStemmer('english', ignore_stopwords = True)
    stemmed_words = [stemmer.stem(word) for word in words]
    
    review_text = " ".join(stemmed_words)
    
    # Return a list of words
    return(review_text)

# Define the model prediction function and return display here
def cluster_display_function(text_values):
    #Clean the text
    vect_text = review_to_wordlist(text_values, remove_stopwords=True)
    if input_text is not None:
        st.write('Progress: returning your cleaned up text...')
    st.write(review_to_wordlist(text_values, remove_stopwords=True))
    # I don't understand why, but I need to basically have a corpus of data to fit and then predict the model, so the pickle is almost useless...
    postfeatures = []
    for i in clusters_df['list_text']:
        postfeatures.append(review_to_wordlist(i, remove_stopwords=True))
    y =  clusters_df['clusters']
    with open("./data/vectorizer.pkl", 'rb') as picklefile:
        vectorizer  = pickle.load(picklefile)
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(postfeatures)
    #Next, fit the text into the classifier model
    with open("./data/class_model.pkl", 'rb') as picklefile:
        model  = pickle.load(picklefile)
    model.fit(X, y)
    #Append the vect_text into the postfeatures space and then use the model predict on it - call the last entry
    #Drop word if it does not appear in current postfeatures list (otherwise the model gets messed up)
    processing_text = vect_text.split()
    word_list = [word for sentence in postfeatures for word in sentence.split()]
    processing_text = [word for word in processing_text if word in word_list]
    processing_text = ' '.join(processing_text)
    postfeatures.append(processing_text)
    vector_text = vectorizer.fit_transform(postfeatures)
    preds = model.predict(vector_text)
    preds_df = pd.DataFrame({'Postfeatures': postfeatures, 'Prediction': preds})
    txt_df = pd.DataFrame(data = {'Input Text' : text_values}, index=[0])
    txt_df['Cluster'] = preds[-1]
    st.write(txt_df)
    if preds[-1] == 0:
        st.header('''Russian Movements and Activities''')
        st.write("""Cluster 0: Russian Movements and Activities. This cluster is focused on Russian movements, troops, and vehicles, inside Ukraine, Russia, and Belarus. This cluster is predictive in the long-term, as announcements or observations of movement are reported on early on, and again reported once the movement has taken place, typically at least a week out. This cluster is helpful to look at as a precursor to offenses, as spikes in activities are associated with Russian mobilizations and deployments.""")
        st.image('https://i.imgur.com/xqq4thv.png', caption = 'This is a generated word cloud image of the most-used words for this cluster, based on a training set of 10,000 posts')
        st.subheader("""Social media posts per day in this cluster""")        
        fig = go.Figure([go.Scatter(x=activity_counts_df['cal_date'], y=activity_counts_df['Cluster_0'])])
        st.plotly_chart(fig, use_container_width = True)
    elif preds[-1] == 1:
        st.header('''Global Russian and Ukrainian Activities''')
        st.write("""Cluster 1: Global Russian and Ukrainian Activities. This cluster has the most amount of posts in it, has a much more global dispersal, and is more generalizable, focused on Russian and Ukrainian-related activities writ-large. This cluster's activity has stayed relatively consistent since November 2022, at a lower level than previously. It will be interesting if other clusters evolve to envelop more of the data as the conflict continues.""")
        st.image('https://i.imgur.com/Envz47G.png', caption = 'This is a generated word cloud image of the most-used words for this cluster, based on a training set of 10,000 posts')
        st.subheader("""Social media posts per day in this cluster""")        
        fig = go.Figure([go.Scatter(x=activity_counts_df['cal_date'], y=activity_counts_df['Cluster_1'])])
        st.plotly_chart(fig, use_container_width = True)
    elif preds[-1] == 2:
        st.header('''The Siege of Mariupol''')
        st.write("""Cluster 2: The Siege of Mariupol. This cluster is directly related to activities around the siege of the city of Mariupol. It is geographically focused on the city, and the timeline of events back it up. The siege was initiated early on in the conflict, was reported on as the city was bombed and assaulted by Russian soldiers, and then eventually activity lulled when the Ukrainian soldiers surrendered. Recently activity spiked due to Russian President Vladimir Putin visiting the city in March 2023. There is predicted to be little activity in this cluster, though if activity in this cluster picks up it might be indicative of a Ukrainian push to retake the city.""")
        st.image('https://i.imgur.com/MTDq9M2.png', caption = 'This is a generated word cloud image of the most-used words for this cluster, based on a training set of 10,000 posts')
        fig = go.Figure([go.Scatter(x=activity_counts_df['cal_date'], y=activity_counts_df['Cluster_2'])])
        st.subheader("""Social media posts per day in this cluster""")        
        st.plotly_chart(fig, use_container_width = True)
    elif preds[-1] == 3:
        st.header('''The Destruction Cluster''')
        st.write("""Cluster 3: The Destruction Cluster. This cluster is focused on destruction wrought by both Russia and Ukraine, as the geographic locations, presumably locations of shelling and other attacks, of these activities are contained largely within Ukraine and Russia. This cluster seems to be related to offenses taken by either side, as well as lulls in fighting as artillery and materiel supplies dwindle. There has recently been an increase in activity in this cluster, as Russia had initiated an offensive in the Donbas region of Ukraine.""")
        st.image('https://i.imgur.com/cehnMGs.png', caption = 'This is a generated word cloud image of the most-used words for this cluster, based on a training set of 10,000 posts')
        st.subheader("""Social media posts per day in this cluster""")        
        fig = go.Figure([go.Scatter(x=activity_counts_df['cal_date'], y=activity_counts_df['Cluster_3'])])
        st.plotly_chart(fig, use_container_width = True)
    elif preds[-1] == 4:
        st.header('''Ukrainian Positions and Activities''')
        st.write("""Cluster 4: Ukrainian Positions and Activities. This cluster is focused on what Ukraine is doing in its battle plans. Activities related to troop movements, positioning, and UAV flights. Geographically the locations of the activities are focused in Eastern Ukraine, as there is a focus on defending the Donbas region. Previous spikes in activity have occurred when Ukraine was moving troops to retake parts of the country. Notably, this cluster has seen a significant decrease in activity recently, but very likely any increase in activity in this cluster means a possible Ukrainian counteroffensive.""")
        st.image('https://i.imgur.com/oSzS78r.png', caption = 'This is a generated word cloud image of the most-used words for this cluster, based on a training set of 10,000 posts')
        st.subheader("""Social media posts per day in this cluster""")        
        fig = go.Figure([go.Scatter(x=activity_counts_df['cal_date'], y=activity_counts_df['Cluster_4'])])
        st.plotly_chart(fig, use_container_width = True)
    elif preds[-1] == 5:
        st.header('''Battle for Bakhmut''')
        st.write("""Cluster 5: Battle for Bakhmut. This cluster is focused on the Donbas, specifically the Donetsk Oblast in Ukraine. It is primarily concerned with activities around the besieged city, which has seen intense fighting since the late Fall, as Russia has focused its efforts on taking this city. It has very recently seen a massive spike in activity, as Russia's offsensive and Ukraine's fierce defense has resulted in the most active part of the conflict right now. Continued increase in this cluster is going to be indicative of intense fighting for the city. If this cluster decreases, it likely means the city has been taken over or is on the verge of being taken over by either side.""")
        st.image('https://i.imgur.com/CmJflm4.png', caption = 'This is a generated word cloud image of the most-used words for this cluster, based on a training set of 10,000 posts')
        st.subheader("""Social media posts per day in this cluster""")        
        fig = go.Figure([go.Scatter(x=activity_counts_df['cal_date'], y=activity_counts_df['Cluster_5'])])
        st.plotly_chart(fig, use_container_width = True)
    elif preds[-1] == 6:
        st.header('''Satellite Imagery''')
        st.write("""Cluster 6: Satellite Imagery. This cluster is focused all over Ukraine and in neighboring countries, and is associated with images and reporting that uses satellite imagery. There have been noteworthy spikes in this cluster since the conflict began, and seems to be indicative of potential uncovering of human rights abuses and atrocities. For example, there was a spike of activity in April 2022, when satellite imagery was used to uncover human rights abuses in the city of Bucha outside Kyiv. This was again repeated in November of 2022, when Ukraine concluded its counteroffensive and took back territory in Kherson and Kharkiv, and again satellite imagery helped uncover evidence of atrocities in the towns formerly controlled by the Russian forces. There has been a lull in this cluster recently, but any increase in activity in this cluster might be indicative of using satellite imagery to document cases of activities that journalists or people keyed in to social media cannot get to.""")
        st.image('https://i.imgur.com/vv9QClc.png', caption = 'This is a generated word cloud image of the most-used words for this cluster, based on a training set of 10,000 posts')
        st.subheader("""Social media posts per day in this cluster""")
        fig = go.Figure([go.Scatter(x=activity_counts_df['cal_date'], y=activity_counts_df['Cluster_6'])])
        fig.update_traces(mode="markers+lines", hovertemplate='Date: %{x} <br>Number of posts: %{y} <extra></extra>')
        fig.add_annotation(x='Apr 13, 2022', y=23,
            text="Russia retreated from <br> north and northeast Ukraine",
            showarrow=True)
        fig.add_annotation(x='Jan 5, 2023', y=31,
            text="Imagery uncovered human rights abuses <br> in areas Ukraine retook",
            showarrow=True)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width = True)
        #Plot number of posts by cluster
    plt.figure(figsize=(10,10))
    preds_df.groupby('Prediction').size().sort_values(ascending=False).plot.bar()
    plt.title('Breakdown of Clusters')
    plt.xticks(rotation=0)
    plt.xlabel("Cluster number")
    plt.ylabel("Number of posts")
    st.pyplot() 
    
       





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
    expander  = st.expander('Expand for problems to address')
    expander.markdown("- The data are not made available in a one-stop-shop location, and require data science expertise to gather, extract-transfer-load, and store.")
    expander.markdown("- The way the data are collected and labeled is not standardized across different sources, and need to be standardized in a more digestable format.")
    expander.markdown("- The data and its contents can be analyzed more holistically, across space and time, and models built out of it.")
    expander.markdown("- The data should be displayed in an understandable, relevant, and useful format, such as through charts and maps, on an interactive website dedicated to this.")

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

elif page == 'Predicting':
    st.subheader('''This page allows you to enter in text of a tweet and have it be clustered by the model.''')
    st.write('''Enter your text in the box below''')
    input_text = st.text_input(label = 'Enter your text here', help = 'Make sure to use English words or translations.')
    if st.button('Results'):
        cluster_display_function(input_text)