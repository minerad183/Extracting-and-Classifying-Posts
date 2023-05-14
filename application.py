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
# import boto3
# import boto3.session

# cred = boto3.Session().get_credentials()
# ACCESS_KEY = cred.access_key
# SECRET_KEY = cred.secret_key
# SESSION_TOKEN = cred.token  ## optional

# st.set_option('deprecation.showPyplotGlobalUse', False)

# s3client = boto3.client('s3', 
#                         aws_access_key_id = ACCESS_KEY, 
#                         aws_secret_access_key = SECRET_KEY, 
#                         aws_session_token = SESSION_TOKEN
#                        )

#Read in data
clusters_df = pd.read_csv('https://clusteringappdevfolder.s3.amazonaws.com/data/get_geoconfirmed_data_clusters.csv', encoding="utf-8")
clusters_df['geometry'] = clusters_df.apply(lambda row: shapely.Point(row['longitude'], row['latitude']), axis=1)
clusters_df = gpd.GeoDataFrame(clusters_df, geometry='geometry')
clusters_df = clusters_df.set_crs("EPSG:4326", allow_override=True)
activity_counts_df = pd.read_csv('https://clusteringappdevfolder.s3.amazonaws.com/data/activity_counts.csv', encoding="utf-8")
activity_counts_df.reset_index(inplace = True)
Ukraine_bnd = gpd.read_file('https://github.com/wmgeolab/geoBoundaries/raw/905b0ba/releaseData/gbOpen/UKR/ADM0/geoBoundaries-UKR-ADM0_simplified.geojson')

# mapbox token
mapboxtoken = 'pk.eyJ1IjoibWluZXJhZCIsImEiOiJjbGhnYm5nZGEwM2JjM3FwbjBnbnN4cHQ4In0.r_9syT8uhNvtxeH4unVpRg'

# # set GeoJSON file path
# path = '/data/geojson.json'
# # write GeoJSON to file
# clusters_df.to_file(path, driver = "GeoJSON", encoding='utf-8')
# with open(path, encoding = 'utf-8') as geofile:
#    j_file = json.load(geofile)
# # index geojson
# i=1
# for feature in j_file["features"]:
#    feature ['id'] = str(i).zfill(2)
#    i += 1


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

# Define a bollinger band requirement for clustered line graphs: https://albertum.medium.com/plotting-bollinger-bands-with-plotly-graph-objects-1c7172899542
def clustered_bollinger_bands(cluster_number):
    WINDOW = 21 #This window size accounts for three weeks of activity. Generally this would be the development of the "news cycle" that the social media posts are referencing
    if cluster_number == 0:
        activity_counts_df['sma'] = activity_counts_df['Cluster_0'].rolling(WINDOW).mean()
        activity_counts_df['std'] = activity_counts_df['Cluster_0'].rolling(WINDOW).std(ddof = 0)
    elif cluster_number == 1:
            activity_counts_df['sma'] = activity_counts_df['Cluster_1'].rolling(WINDOW).mean()
            activity_counts_df['std'] = activity_counts_df['Cluster_1'].rolling(WINDOW).std(ddof = 0)
    elif cluster_number == 2:
            activity_counts_df['sma'] = activity_counts_df['Cluster_2'].rolling(WINDOW).mean()
            activity_counts_df['std'] = activity_counts_df['Cluster_3'].rolling(WINDOW).std(ddof = 0)
    elif cluster_number == 3:
            activity_counts_df['sma'] = activity_counts_df['Cluster_3'].rolling(WINDOW).mean()
            activity_counts_df['std'] = activity_counts_df['Cluster_3'].rolling(WINDOW).std(ddof = 0)
    elif cluster_number == 4:
            activity_counts_df['sma'] = activity_counts_df['Cluster_4'].rolling(WINDOW).mean()
            activity_counts_df['std'] = activity_counts_df['Cluster_4'].rolling(WINDOW).std(ddof = 0)
    elif cluster_number == 5:
            activity_counts_df['sma'] = activity_counts_df['Cluster_5'].rolling(WINDOW).mean()
            activity_counts_df['std'] = activity_counts_df['Cluster_5'].rolling(WINDOW).std(ddof = 0)
    elif cluster_number == 6:
            activity_counts_df['sma'] = activity_counts_df['Cluster_6'].rolling(WINDOW).mean()
            activity_counts_df['std'] = activity_counts_df['Cluster_6'].rolling(WINDOW).std(ddof = 0)

#Process for the centroids map. This is in the app because it could be updated on the fly with new data
ukr_list = []

for i in range(0, 7):
    clusters_df['geometry'] = clusters_df['geometry'].to_crs('EPSG:20007') # https://spatialreference.org/ref/epsg/20007/
    new_row = clusters_df[clusters_df['clusters'] == i].unary_union.centroid
    print(new_row)
    ukr_list.append(new_row)

ukr_centroids = pd.DataFrame({'geometry' : ukr_list})
ukr_centroids = gpd.GeoDataFrame(ukr_centroids, geometry='geometry')
ukr_centroids['geometry'] = ukr_centroids['geometry'].set_crs('EPSG:20007')
ukr_centroids['geometry'] = ukr_centroids['geometry'].to_crs(crs = 'EPSG:4326')
ukr_centroids['lat'] = ukr_centroids['geometry'].apply(lambda geom: geom.y)
ukr_centroids['lon'] = ukr_centroids['geometry'].apply(lambda geom: geom.x)

ukr_centroids['clust'] = [i for i in range(0, 7)]
# Center of Mass setup
def center_of_mass_plot(cluster_number):
        marker_colors = {0: 'rgb(255, 195, 127)', 1: 'rgb(127, 179, 228)',
                        2: 'rgb(190, 247, 208)', 3: 'rgb(255, 213, 213)',
                        4: 'rgb(193, 228, 255)', 5: 'rgb(255, 149, 149)',
                        6: 'rgb(148, 215, 206)'}
        layout_com = dict(title_text ='Center of Mass Map of this Cluster', title_x =0.5,  
            width=950, height=700,mapbox = dict(center= dict(lat=47,  
            lon=35), accesstoken= mapboxtoken, zoom=4,style="carto-positron"))
        traces = []
        Lat = []
        Lon = []
        Clust = []
        for index, row in ukr_centroids.iterrows():
            lat = row['lat']
            lon = row['lon']
            clust = row['clust']

            if row['clust'] == cluster_number:
                marker_color = marker_colors.get(cluster_number)
            else:
                marker_color = 'rgb(128,128,128)'
            
            Lat.append(lat)
            Lon.append(lon)
            traces.append(marker_color)
            Clust.append(clust)
        
        trace = go.Scattermapbox(showlegend = True, lat = Lat, lon = Lon, mode = 'markers', 
                                marker = dict(size = 12, color = traces), name = f'Cluster {cluster_number}', hovertemplate='Latitude: %{lat} <br> Longitude: %{lon} <extra></extra>')
        fig = go.Figure(data = trace, layout = layout_com)
        st.plotly_chart(fig, use_container_width = True)
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
    # response = s3client.get_object(Bucket='clusteringappdevfolder', Key='https://clusteringappdevfolder.s3.amazonaws.com/data/vectorizer.pkl')
    # body = response['Body'].read()
    # vectorizer = pickle.loads(body)
    model = pd.read_pickle('https://clusteringappdevfolder.s3.amazonaws.com/data/vectorizer.pkl')    
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(postfeatures)
    #Next, fit the text into the classifier model
    # response = s3client.get_object(Bucket='clusteringappdevfolder', Key='https://clusteringappdevfolder.s3.amazonaws.com/data/class_model.pkl')
    # body = response['Body'].read()
    # model = pickle.loads(body)
    model = pd.read_pickle('https://clusteringappdevfolder.s3.amazonaws.com/data/class_model.pkl')
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
   
    #Set up graphs and frames that get called when these clusters are returned
    if preds[-1] == 0:
        st.header('''Russian Movements and Activities''')
        st.write("""Cluster 0: Russian Movements and Activities. This cluster is focused on Russian movements, troops, and vehicles, inside Ukraine, Russia, and Belarus. This cluster is predictive in the long-term, as announcements or observations of movement are reported on early on, and again reported once the movement has taken place, typically at least a week out. This cluster is helpful to look at as a precursor to offenses, as spikes in activities are associated with Russian mobilizations and deployments.""")
        st.image('https://i.imgur.com/xqq4thv.png', caption = 'This is a generated word cloud image of the most-used words for this cluster, based on a training set of 10,000 posts')
        # Line Graph of activity over time with Bollinger Bands
        fig = go.Figure([go.Scatter(x=activity_counts_df['cal_date'], y=activity_counts_df['Cluster_0'])])
        fig.update_traces(mode="markers+lines", hovertemplate='Date: %{x} <br>Number of posts: %{y} <extra></extra>')
        clustered_bollinger_bands(0)
        fig.add_trace(go.Scatter(x = activity_counts_df['cal_date'], y = activity_counts_df['sma'] + (activity_counts_df['std'] * 2), line_color = 'gray', line = {'dash': 'dash'}, name = 'Upper Band', opacity = 0.3))
        fig.add_trace(go.Scatter(x = activity_counts_df['cal_date'], y = activity_counts_df['sma'] - activity_counts_df['sma'], name = 'Lower Band', opacity = 0.3, fill = 'tonexty', line_color = 'gray', line = {'dash': 'dash'}))
        fig.add_annotation(x=activity_counts_df.loc[activity_counts_df['cal_date'] == '5/05/2023'].to_json(), y=31,
            text="Shaded areas identify number of social media posts in an expected 21-day average. <br> Anything not within this band signifies anomalous activity and may be indicative of a newsworthy development.",
            showarrow=True)        
        fig.update_layout(showlegend=False, title = "Social media posts per day in this cluster") #gets too cluttered with the legend, and the captions and annotations explain it
        st.plotly_chart(fig, use_container_width = True)
        st.write("""Spikes in this cluster's post activity refer to the initial Russian
        invasion in March-April 2022, mobilization announcements in September 2022, and tracking of Russia's 2023 offensive.""")        
        #"Center of Mass" Map, where the geographic center of the cluster is
        center_of_mass_plot(0)
        st.write("""This map shows the average location of the social media posts in this cluster.
        This cluster has its origins in the heart of the current conflict front. Notably it is being pulled westward because of Russia's initial invasion as well as social media coverage of troop movements within Russia and Belarus.""")
    elif preds[-1] == 1:
        st.header('''Global Russian and Ukrainian Activities''')
        st.write("""Cluster 1: Global Russian and Ukrainian Activities. This cluster has the most amount of posts in it, has a much more global dispersal, and is more generalizable, focused on Russian and Ukrainian-related activities writ-large. This cluster's activity has stayed relatively consistent since November 2022, at a lower level than previously. It will be interesting if other clusters evolve to envelop more of the data as the conflict continues.""")
        st.image('https://i.imgur.com/Envz47G.png', caption = 'This is a generated word cloud image of the most-used words for this cluster, based on a training set of 10,000 posts')       
        fig = go.Figure([go.Scatter(x=activity_counts_df['cal_date'], y=activity_counts_df['Cluster_1'])])
        fig.update_traces(mode="markers+lines", hovertemplate='Date: %{x} <br>Number of posts: %{y} <extra></extra>')
        clustered_bollinger_bands(1)
        fig.add_trace(go.Scatter(x = activity_counts_df['cal_date'], y = activity_counts_df['sma'] + (activity_counts_df['std'] * 2), line_color = 'gray', line = {'dash': 'dash'}, name = 'Upper Band', opacity = 0.3))
        fig.add_trace(go.Scatter(x = activity_counts_df['cal_date'], y = activity_counts_df['sma'] - activity_counts_df['sma'], name = 'Lower Band', opacity = 0.3, fill = 'tonexty', line_color = 'gray', line = {'dash': 'dash'}))
        fig.add_annotation(x=activity_counts_df.loc[activity_counts_df['cal_date'] == '5/05/2023'].to_json(), y=55,
            text="Shaded areas identify number of social media posts in an expected 21-day average. <br> Anything not within this band signifies anomalous activity and may be indicative of a newsworthy development.",
            showarrow=True)        
        fig.update_layout(showlegend=False, title = "Social media posts per day in this cluster")
        st.plotly_chart(fig, use_container_width = True)
        st.write("""This cluster has spiked in the past with major newsworthy information, such as Ukraine's counteroffensives.""")        
        center_of_mass_plot(1)
        st.write("""This map shows the average location of the social media posts in this cluster.
        This cluster is located very close to the current front, but more northward as events are contained within this cluster from inside of Russia.""")
    elif preds[-1] == 2:
        st.header('''The Siege of Mariupol''')
        st.write("""Cluster 2: The Siege of Mariupol. This cluster is directly related to activities around the siege of the city of Mariupol. It is geographically focused on the city, and the timeline of events back it up. The siege was initiated early on in the conflict, was reported on as the city was bombed and assaulted by Russian soldiers, and then eventually activity lulled when the Ukrainian soldiers surrendered. Recently activity spiked due to Russian President Vladimir Putin visiting the city in March 2023. There is predicted to be little activity in this cluster, though if activity in this cluster picks up it might be indicative of a Ukrainian push to retake the city.""")
        st.image('https://i.imgur.com/MTDq9M2.png', caption = 'This is a generated word cloud image of the most-used words for this cluster, based on a training set of 10,000 posts')    
        fig = go.Figure([go.Scatter(x=activity_counts_df['cal_date'], y=activity_counts_df['Cluster_2'])])
        fig.update_traces(mode="markers+lines", hovertemplate='Date: %{x} <br>Number of posts: %{y} <extra></extra>')
        clustered_bollinger_bands(2)
        fig.add_trace(go.Scatter(x = activity_counts_df['cal_date'], y = activity_counts_df['sma'] + (activity_counts_df['std'] * 2), line_color = 'gray', line = {'dash': 'dash'}, name = 'Upper Band', opacity = 0.3))
        fig.add_trace(go.Scatter(x = activity_counts_df['cal_date'], y = activity_counts_df['sma'] - activity_counts_df['sma'], name = 'Lower Band', opacity = 0.3, fill = 'tonexty', line_color = 'gray', line = {'dash': 'dash'}))
        fig.add_annotation(x=activity_counts_df.loc[activity_counts_df['cal_date'] == '5/05/2023'].to_json(), y=31,
            text="Shaded areas identify number of social media posts in an expected 21-day average. <br> Anything not within this band signifies anomalous activity and may be indicative of a newsworthy development.",
            showarrow=True)        
        fig.update_layout(showlegend=False, title = "Social media posts per day in this cluster")
        st.plotly_chart(fig, use_container_width = True)
        st.write("""This cluster had major activity early in the conflict as Russia besieged the city on the coast of the Black Sea.
        It again saw activity in March of 2023 as posts referencing Russian President Vladimir Putin's trip to the city.""")        
        center_of_mass_plot(2)
        st.write("""This map shows the average location of the social media posts in this cluster.
        The location is almost directly focused on the city of Mariupol, as it was one of the most noteworthy cities to cover in the early phase of the conflict.""")
    elif preds[-1] == 3:
        st.header('''The Destruction Cluster''')
        st.write("""Cluster 3: The Destruction Cluster. This cluster is focused on destruction wrought by both Russia and Ukraine, as the geographic locations, presumably locations of shelling and other attacks, of these activities are contained largely within Ukraine and Russia. This cluster seems to be related to offenses taken by either side, as well as lulls in fighting as artillery and materiel supplies dwindle. There has recently been an increase in activity in this cluster, as Russia had initiated an offensive in the Donbas region of Ukraine.""")
        st.image('https://i.imgur.com/cehnMGs.png', caption = 'This is a generated word cloud image of the most-used words for this cluster, based on a training set of 10,000 posts')       
        fig = go.Figure([go.Scatter(x=activity_counts_df['cal_date'], y=activity_counts_df['Cluster_3'])])
        fig.update_traces(mode="markers+lines", hovertemplate='Date: %{x} <br>Number of posts: %{y} <extra></extra>')
        clustered_bollinger_bands(3)
        fig.add_trace(go.Scatter(x = activity_counts_df['cal_date'], y = activity_counts_df['sma'] + (activity_counts_df['std'] * 2), line_color = 'gray', line = {'dash': 'dash'}, name = 'Upper Band', opacity = 0.3))
        fig.add_trace(go.Scatter(x = activity_counts_df['cal_date'], y = activity_counts_df['sma'] - activity_counts_df['sma'], name = 'Lower Band', opacity = 0.3, fill = 'tonexty', line_color = 'gray', line = {'dash': 'dash'}))
        fig.add_annotation(x=activity_counts_df.loc[activity_counts_df['cal_date'] == '5/05/2023'].to_json(), y=31,
            text="Shaded areas identify number of social media posts in an expected 21-day average. <br> Anything not within this band signifies anomalous activity and may be indicative of a newsworthy development.",
            showarrow=True)        
        fig.update_layout(showlegend=False, title = "Social media posts per day in this cluster")        
        st.plotly_chart(fig, use_container_width = True)
        st.write("""This cluster is indicative of social media activity reporting about destructive actions in both Ukraine and Russia. It has stayed relatively stable since dropping in late-May, 2022.""")
        center_of_mass_plot(3)
        st.write("""This map shows the average location of the social media posts in this cluster.
        This location is focused in central-east Ukraine, as many destruction-related events have been published around the conflict's front, plus in the nearby Russian province of Belgorod.""")
    elif preds[-1] == 4:
        st.header('''Ukrainian Positions and Activities''')
        st.write("""Cluster 4: Ukrainian Positions and Activities. This cluster is focused on what Ukraine is doing in its battle plans. Activities related to troop movements, positioning, and UAV flights. Geographically the locations of the activities are focused in Eastern Ukraine, as there is a focus on defending the Donbas region. Previous spikes in activity have occurred when Ukraine was moving troops to retake parts of the country. Notably, this cluster has seen a significant decrease in activity recently, but very likely any increase in activity in this cluster means a possible Ukrainian counteroffensive.""")
        st.image('https://i.imgur.com/oSzS78r.png', caption = 'This is a generated word cloud image of the most-used words for this cluster, based on a training set of 10,000 posts') 
        fig = go.Figure([go.Scatter(x=activity_counts_df['cal_date'], y=activity_counts_df['Cluster_4'])])
        fig.update_traces(mode="markers+lines", hovertemplate='Date: %{x} <br>Number of posts: %{y} <extra></extra>')
        clustered_bollinger_bands(4)
        fig.add_trace(go.Scatter(x = activity_counts_df['cal_date'], y = activity_counts_df['sma'] + (activity_counts_df['std'] * 2), line_color = 'gray', line = {'dash': 'dash'}, name = 'Upper Band', opacity = 0.3))
        fig.add_trace(go.Scatter(x = activity_counts_df['cal_date'], y = activity_counts_df['sma'] - activity_counts_df['sma'], name = 'Lower Band', opacity = 0.3, fill = 'tonexty', line_color = 'gray', line = {'dash': 'dash'}))
        fig.add_annotation(x=activity_counts_df.loc[activity_counts_df['cal_date'] == '5/05/2023'].to_json(), y=31,
            text="Shaded areas identify number of social media posts in an expected 21-day average. <br> Anything not within this band signifies anomalous activity and may be indicative of a newsworthy development.",
            showarrow=True)        
        fig.update_layout(showlegend=False, title = "Social media posts per day in this cluster")        
        st.plotly_chart(fig, use_container_width = True)
        st.write("""Spikes in this cluster's post activity refer to the Ukrainian movements to retake lost territorial control. A spike in this cluster may be indicative of news of a Ukrainian counteroffensive.""")                
        center_of_mass_plot(4)
        st.write("""This map shows the average location of the social media posts in this cluster.
        The location is nearby the current conflict's front, notably behind the Ukrainian lines, as this cluster is associated with the side's activities.""")
    elif preds[-1] == 5:
        st.header('''Battle for Bakhmut''')
        st.write("""Cluster 5: Battle for Bakhmut. This cluster is focused on the Donbas, specifically the Donetsk Oblast in Ukraine. It is primarily concerned with activities around the besieged city, which has seen intense fighting since the late Fall, as Russia has focused its efforts on taking this city. It has very recently seen a massive spike in activity, as Russia's offsensive and Ukraine's fierce defense has resulted in the most active part of the conflict right now. Continued increase in this cluster is going to be indicative of intense fighting for the city. If this cluster decreases, it likely means the city has been taken over or is on the verge of being taken over by either side.""")
        st.image('https://i.imgur.com/CmJflm4.png', caption = 'This is a generated word cloud image of the most-used words for this cluster, based on a training set of 10,000 posts')     
        fig = go.Figure([go.Scatter(x=activity_counts_df['cal_date'], y=activity_counts_df['Cluster_5'])])
        fig.update_traces(mode="markers+lines", hovertemplate='Date: %{x} <br>Number of posts: %{y} <extra></extra>')        
        clustered_bollinger_bands(5)
        fig.add_trace(go.Scatter(x = activity_counts_df['cal_date'], y = activity_counts_df['sma'] + (activity_counts_df['std'] * 2), line_color = 'gray', line = {'dash': 'dash'}, name = 'Upper Band', opacity = 0.3))
        fig.add_trace(go.Scatter(x = activity_counts_df['cal_date'], y = activity_counts_df['sma'] - activity_counts_df['sma'], name = 'Lower Band', opacity = 0.3, fill = 'tonexty', line_color = 'gray', line = {'dash': 'dash'}))
        fig.add_annotation(x=activity_counts_df.loc[activity_counts_df['cal_date'] == '5/05/2023'].to_json(), y=42,
            text="Shaded areas identify number of social media posts in an expected 21-day average. <br> Anything not within this band signifies anomalous activity and may be indicative of a newsworthy development.",
            showarrow=True)   
        fig.update_layout(showlegend=False, title = "Social media posts per day in this cluster")             
        st.plotly_chart(fig, use_container_width = True)
        st.write("""This cluster has seen a massive spike in activity as social media posts about the fight in the Donbas,
        specifically around the city of Bakhmut. """)
        center_of_mass_plot(5)
        st.write("""This map shows the average location of the social media posts in this cluster.
        The location is nearby the city of Bakhmut, where fighting has been most intense as of April 2023.""")
    elif preds[-1] == 6:
        st.header('''Satellite Imagery''')
        st.write("""Cluster 6: Satellite Imagery. This cluster is focused all over Ukraine and in neighboring countries, and is associated with images and reporting that uses satellite imagery. There have been noteworthy spikes in this cluster since the conflict began, and seems to be indicative of potential uncovering of human rights abuses and atrocities. For example, there was a spike of activity in April 2022, when satellite imagery was used to uncover human rights abuses in the city of Bucha outside Kyiv. This was again repeated in November of 2022, when Ukraine concluded its counteroffensive and took back territory in Kherson and Kharkiv, and again satellite imagery helped uncover evidence of atrocities in the towns formerly controlled by the Russian forces. There has been a lull in this cluster recently, but any increase in activity in this cluster might be indicative of using satellite imagery to document cases of activities that journalists or people keyed in to social media cannot get to.""")
        st.image('https://i.imgur.com/vv9QClc.png', caption = 'This is a generated word cloud image of the most-used words for this cluster, based on a training set of 10,000 posts')
        fig = go.Figure([go.Scatter(x=activity_counts_df['cal_date'], y=activity_counts_df['Cluster_6'])])
        fig.update_traces(mode="markers+lines", hovertemplate='Date: %{x} <br>Number of posts: %{y} <extra></extra>')
        clustered_bollinger_bands(6)
        fig.add_trace(go.Scatter(x = activity_counts_df['cal_date'], y = activity_counts_df['sma'] + (activity_counts_df['std'] * 2), line_color = 'gray', line = {'dash': 'dash'}, name = 'Upper Band', opacity = 0.3))
        fig.add_trace(go.Scatter(x = activity_counts_df['cal_date'], y = activity_counts_df['sma'] - activity_counts_df['sma'], name = 'Lower Band', opacity = 0.3, fill = 'tonexty', line_color = 'gray', line = {'dash': 'dash'}))
        fig.add_annotation(x=activity_counts_df.loc[activity_counts_df['cal_date'] == '5/05/2023'].to_json(), y=31, #I couldn't get annotations to work exactly how I wanted, so I'm pivoting
            text="Shaded areas identify number of social media posts in an expected 21-day average. <br> Anything not within this band signifies anomalous activity and may be indicative of a newsworthy development.",
            showarrow=True)
        fig.update_layout(showlegend=False, title = "Social media posts per day in this cluster")
        st.plotly_chart(fig, use_container_width = True)
        st.write("""This cluster's activity spikes when satellite imagery is used in social media posts.
        Notably, satellite imagery has been used to augment coverage of human rights abuses in Ukraine, and spikes in this cluster have focused on these posts.""")
        center_of_mass_plot(6)
        st.write("""This map shows the average location of the social media posts in this cluster.
        It is more centrally located, as many of the posts are referencing activities along the conflict's front and along the coast of Ukraine.""")        
    #Plot number of posts by cluster
    st.subheader('''Here is the breakdown of the number of posts in each cluster''')
    plt.figure(figsize=(10,10))
    preds_df.groupby('Prediction').size().sort_values(ascending=False).plot.bar()
    plt.title('Breakdown of Clusters')
    plt.xticks(rotation=0)
    plt.xlabel("Cluster number")
    plt.ylabel("Number of posts")
    st.pyplot() 
    
#I used these sites to help make the interactive map: https://towardsdatascience.com/how-to-create-interactive-map-plots-with-plotly-7b57e889239a
# https://towardsdatascience.com/build-a-multi-layer-map-using-streamlit-2b4d44eb28f3
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
        
        # drop-down 2: select type of event to visualize
        dict(
            buttons=list([
                dict(label = 'All Clusters',
                    method = 'update',
                    args = [{'visible': [True, True, True, True, True, True, True]}]),
                dict(label = 'Russian Movements and Activities',
                    method = 'update',
                    args = [{'visible': [False, False, False, False, False, False, True]}]),
                dict(label = 'Global Russian and Ukrainian Activities',
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
        change_variable = 'visible' 
    st.text("")
    st.text("")
    st.text("")
    option = st.selectbox(label = 'Select another cluster to view',
                    options = ('Russian Movements and Activities',
                            'Global Russian and Ukrainian Activities',
                            'The Siege of Mariupol',
                            'The Destruction Cluster',
                            'Ukrainian Positions and Activities',
                            'Battle for Bakhmut',
                            'Satellite Imagery'))
    if option == 'Russian Movements and Activities':
        st.header('''Russian Movements and Activities''')
        st.write("""Cluster 0: Russian Movements and Activities. This cluster is focused on Russian movements, troops, and vehicles, inside Ukraine, Russia, and Belarus. This cluster is predictive in the long-term, as announcements or observations of movement are reported on early on, and again reported once the movement has taken place, typically at least a week out. This cluster is helpful to look at as a precursor to offenses, as spikes in activities are associated with Russian mobilizations and deployments.""")
        st.image('https://i.imgur.com/xqq4thv.png', caption = 'This is a generated word cloud image of the most-used words for this cluster, based on a training set of 10,000 posts')
        # Line Graph of activity over time with Bollinger Bands
        fig = go.Figure([go.Scatter(x=activity_counts_df['cal_date'], y=activity_counts_df['Cluster_0'])])
        fig.update_traces(mode="markers+lines", hovertemplate='Date: %{x} <br>Number of posts: %{y} <extra></extra>')
        clustered_bollinger_bands(0)
        fig.add_trace(go.Scatter(x = activity_counts_df['cal_date'], y = activity_counts_df['sma'] + (activity_counts_df['std'] * 2), line_color = 'gray', line = {'dash': 'dash'}, name = 'Upper Band', opacity = 0.3))
        fig.add_trace(go.Scatter(x = activity_counts_df['cal_date'], y = activity_counts_df['sma'] - activity_counts_df['sma'], name = 'Lower Band', opacity = 0.3, fill = 'tonexty', line_color = 'gray', line = {'dash': 'dash'}))
        fig.add_annotation(x=activity_counts_df.loc[activity_counts_df['cal_date'] == '5/05/2023'].to_json(), y=31,
            text="Shaded areas identify number of social media posts in an expected 21-day average. <br> Anything not within this band signifies anomalous activity and may be indicative of a newsworthy development.",
            showarrow=True)        
        fig.update_layout(showlegend=False, title = "Social media posts per day in this cluster") #gets too cluttered with the legend, and the captions and annotations explain it
        st.plotly_chart(fig, use_container_width = True)
        st.write("""Spikes in this cluster's post activity refer to the initial Russian
        invasion in March-April 2022, mobilization announcements in September 2022, and tracking of Russia's 2023 offensive.""")
        #"Center of Mass" Map, where the geographic center of the cluster is
        center_of_mass_plot(0)
        st.write("""This map shows the average location of the social media posts in this cluster.
        This cluster has its origins in the heart of the current conflict front. Notably it is being pulled westward because of Russia's initial invasion as well as social media coverage of troop movements within Russia and Belarus.""")
    elif option == 'Global Russian and Ukrainian Activities':
        st.header('''Global Russian and Ukrainian Activities''')
        st.write("""Cluster 1: Global Russian and Ukrainian Activities. This cluster has the most amount of posts in it, has a much more global dispersal, and is more generalizable, focused on Russian and Ukrainian-related activities writ-large. This cluster's activity has stayed relatively consistent since November 2022, at a lower level than previously. It will be interesting if other clusters evolve to envelop more of the data as the conflict continues.""")
        st.image('https://i.imgur.com/Envz47G.png', caption = 'This is a generated word cloud image of the most-used words for this cluster, based on a training set of 10,000 posts')       
        fig = go.Figure([go.Scatter(x=activity_counts_df['cal_date'], y=activity_counts_df['Cluster_1'])])
        fig.update_traces(mode="markers+lines", hovertemplate='Date: %{x} <br>Number of posts: %{y} <extra></extra>')
        clustered_bollinger_bands(1)
        fig.add_trace(go.Scatter(x = activity_counts_df['cal_date'], y = activity_counts_df['sma'] + (activity_counts_df['std'] * 2), line_color = 'gray', line = {'dash': 'dash'}, name = 'Upper Band', opacity = 0.3))
        fig.add_trace(go.Scatter(x = activity_counts_df['cal_date'], y = activity_counts_df['sma'] - activity_counts_df['sma'], name = 'Lower Band', opacity = 0.3, fill = 'tonexty', line_color = 'gray', line = {'dash': 'dash'}))
        fig.add_annotation(x=activity_counts_df.loc[activity_counts_df['cal_date'] == '5/05/2023'].to_json(), y=55,
            text="Shaded areas identify number of social media posts in an expected 21-day average. <br> Anything not within this band signifies anomalous activity and may be indicative of a newsworthy development.",
            showarrow=True)        
        fig.update_layout(showlegend=False, title = "Social media posts per day in this cluster")
        st.plotly_chart(fig, use_container_width = True)
        st.write("""This cluster has spiked in the past with major newsworthy information, such as Ukraine's counteroffensives.""")        
        center_of_mass_plot(1)
        st.write("""This map shows the average location of the social media posts in this cluster.
        This cluster is located very close to the current front, but more northward as events are contained within this cluster from inside of Russia.""")
    elif option == 'The Siege of Mariupol':
        st.header('''The Siege of Mariupol''')
        st.write("""Cluster 2: The Siege of Mariupol. This cluster is directly related to activities around the siege of the city of Mariupol. It is geographically focused on the city, and the timeline of events back it up. The siege was initiated early on in the conflict, was reported on as the city was bombed and assaulted by Russian soldiers, and then eventually activity lulled when the Ukrainian soldiers surrendered. Recently activity spiked due to Russian President Vladimir Putin visiting the city in March 2023. There is predicted to be little activity in this cluster, though if activity in this cluster picks up it might be indicative of a Ukrainian push to retake the city.""")
        st.image('https://i.imgur.com/MTDq9M2.png', caption = 'This is a generated word cloud image of the most-used words for this cluster, based on a training set of 10,000 posts')    
        fig = go.Figure([go.Scatter(x=activity_counts_df['cal_date'], y=activity_counts_df['Cluster_2'])])
        fig.update_traces(mode="markers+lines", hovertemplate='Date: %{x} <br>Number of posts: %{y} <extra></extra>')
        clustered_bollinger_bands(2)
        fig.add_trace(go.Scatter(x = activity_counts_df['cal_date'], y = activity_counts_df['sma'] + (activity_counts_df['std'] * 2), line_color = 'gray', line = {'dash': 'dash'}, name = 'Upper Band', opacity = 0.3))
        fig.add_trace(go.Scatter(x = activity_counts_df['cal_date'], y = activity_counts_df['sma'] - activity_counts_df['sma'], name = 'Lower Band', opacity = 0.3, fill = 'tonexty', line_color = 'gray', line = {'dash': 'dash'}))
        fig.add_annotation(x=activity_counts_df.loc[activity_counts_df['cal_date'] == '5/05/2023'].to_json(), y=31,
            text="Shaded areas identify number of social media posts in an expected 21-day average. <br> Anything not within this band signifies anomalous activity and may be indicative of a newsworthy development.",
            showarrow=True)        
        fig.update_layout(showlegend=False, title = "Social media posts per day in this cluster")
        st.plotly_chart(fig, use_container_width = True)
        st.write("""This cluster had major activity early in the conflict as Russia besieged the city on the coast of the Black Sea.
        It again saw activity in March of 2023 as posts referencing Russian President Vladimir Putin's trip to the city.""")
        center_of_mass_plot(2)
        st.write("""This map shows the average location of the social media posts in this cluster.
        The location is almost directly focused on the city of Mariupol, as it was one of the most noteworthy cities to cover in the early phase of the conflict.""")
    elif option == 'The Destruction Cluster':
        st.header('''The Destruction Cluster''')
        st.write("""Cluster 3: The Destruction Cluster. This cluster is focused on destruction wrought by both Russia and Ukraine, as the geographic locations, presumably locations of shelling and other attacks, of these activities are contained largely within Ukraine and Russia. This cluster seems to be related to offenses taken by either side, as well as lulls in fighting as artillery and materiel supplies dwindle. There has recently been an increase in activity in this cluster, as Russia had initiated an offensive in the Donbas region of Ukraine.""")
        st.image('https://i.imgur.com/cehnMGs.png', caption = 'This is a generated word cloud image of the most-used words for this cluster, based on a training set of 10,000 posts')       
        fig = go.Figure([go.Scatter(x=activity_counts_df['cal_date'], y=activity_counts_df['Cluster_3'])])
        fig.update_traces(mode="markers+lines", hovertemplate='Date: %{x} <br>Number of posts: %{y} <extra></extra>')
        clustered_bollinger_bands(3)
        fig.add_trace(go.Scatter(x = activity_counts_df['cal_date'], y = activity_counts_df['sma'] + (activity_counts_df['std'] * 2), line_color = 'gray', line = {'dash': 'dash'}, name = 'Upper Band', opacity = 0.3))
        fig.add_trace(go.Scatter(x = activity_counts_df['cal_date'], y = activity_counts_df['sma'] - activity_counts_df['sma'], name = 'Lower Band', opacity = 0.3, fill = 'tonexty', line_color = 'gray', line = {'dash': 'dash'}))
        fig.add_annotation(x=activity_counts_df.loc[activity_counts_df['cal_date'] == '5/05/2023'].to_json(), y=31,
            text="Shaded areas identify number of social media posts in an expected 21-day average. <br> Anything not within this band signifies anomalous activity and may be indicative of a newsworthy development.",
            showarrow=True)        
        fig.update_layout(showlegend=False, title = "Social media posts per day in this cluster")        
        st.plotly_chart(fig, use_container_width = True)
        st.write("""This cluster is indicative of social media activity reporting about destructive actions in both Ukraine and Russia. It has stayed relatively stable since dropping in late-May, 2022.""")
        center_of_mass_plot(3)
        st.write("""This map shows the average location of the social media posts in this cluster.
        This location is focused in central-east Ukraine, as many destruction-related events have been published around the conflict's front, plus in the nearby Russian province of Belgorod.""")
    elif option == 'Ukrainian Positions and Activities':
        st.header('''Ukrainian Positions and Activities''')
        st.write("""Cluster 4: Ukrainian Positions and Activities. This cluster is focused on what Ukraine is doing in its battle plans. Activities related to troop movements, positioning, and UAV flights. Geographically the locations of the activities are focused in Eastern Ukraine, as there is a focus on defending the Donbas region. Previous spikes in activity have occurred when Ukraine was moving troops to retake parts of the country. Notably, this cluster has seen a significant decrease in activity recently, but very likely any increase in activity in this cluster means a possible Ukrainian counteroffensive.""")
        st.image('https://i.imgur.com/oSzS78r.png', caption = 'This is a generated word cloud image of the most-used words for this cluster, based on a training set of 10,000 posts') 
        fig = go.Figure([go.Scatter(x=activity_counts_df['cal_date'], y=activity_counts_df['Cluster_4'])])
        fig.update_traces(mode="markers+lines", hovertemplate='Date: %{x} <br>Number of posts: %{y} <extra></extra>')
        clustered_bollinger_bands(4)
        fig.add_trace(go.Scatter(x = activity_counts_df['cal_date'], y = activity_counts_df['sma'] + (activity_counts_df['std'] * 2), line_color = 'gray', line = {'dash': 'dash'}, name = 'Upper Band', opacity = 0.3))
        fig.add_trace(go.Scatter(x = activity_counts_df['cal_date'], y = activity_counts_df['sma'] - activity_counts_df['sma'], name = 'Lower Band', opacity = 0.3, fill = 'tonexty', line_color = 'gray', line = {'dash': 'dash'}))
        fig.add_annotation(x=activity_counts_df.loc[activity_counts_df['cal_date'] == '5/05/2023'].to_json(), y=31,
            text="Shaded areas identify number of social media posts in an expected 21-day average. <br> Anything not within this band signifies anomalous activity and may be indicative of a newsworthy development.",
            showarrow=True)        
        fig.update_layout(showlegend=False, title = "Social media posts per day in this cluster")        
        st.plotly_chart(fig, use_container_width = True)
        st.write("""Spikes in this cluster's post activity refer to the Ukrainian movements to retake lost territorial control. A spike in this cluster may be indicative of news of a Ukrainian counteroffensive.""")        
        center_of_mass_plot(4)
        st.write("""This map shows the average location of the social media posts in this cluster.
        The location is nearby the current conflict's front, notably behind the Ukrainian lines, as this cluster is associated with the side's activities.""")
    elif option == 'Battle for Bakhmut':
        st.header('''Battle for Bakhmut''')
        st.write("""Cluster 5: Battle for Bakhmut. This cluster is focused on the Donbas, specifically the Donetsk Oblast in Ukraine. It is primarily concerned with activities around the besieged city, which has seen intense fighting since the late Fall, as Russia has focused its efforts on taking this city. It has very recently seen a massive spike in activity, as Russia's offsensive and Ukraine's fierce defense has resulted in the most active part of the conflict right now. Continued increase in this cluster is going to be indicative of intense fighting for the city. If this cluster decreases, it likely means the city has been taken over or is on the verge of being taken over by either side.""")
        st.image('https://i.imgur.com/CmJflm4.png', caption = 'This is a generated word cloud image of the most-used words for this cluster, based on a training set of 10,000 posts')     
        fig = go.Figure([go.Scatter(x=activity_counts_df['cal_date'], y=activity_counts_df['Cluster_5'])])
        fig.update_traces(mode="markers+lines", hovertemplate='Date: %{x} <br>Number of posts: %{y} <extra></extra>')        
        clustered_bollinger_bands(5)
        fig.add_trace(go.Scatter(x = activity_counts_df['cal_date'], y = activity_counts_df['sma'] + (activity_counts_df['std'] * 2), line_color = 'gray', line = {'dash': 'dash'}, name = 'Upper Band', opacity = 0.3))
        fig.add_trace(go.Scatter(x = activity_counts_df['cal_date'], y = activity_counts_df['sma'] - activity_counts_df['sma'], name = 'Lower Band', opacity = 0.3, fill = 'tonexty', line_color = 'gray', line = {'dash': 'dash'}))
        fig.add_annotation(x=activity_counts_df.loc[activity_counts_df['cal_date'] == '5/05/2023'].to_json(), y=42,
            text="Shaded areas identify number of social media posts in an expected 21-day average. <br> Anything not within this band signifies anomalous activity and may be indicative of a newsworthy development.",
            showarrow=True)   
        fig.update_layout(showlegend=False, title = "Social media posts per day in this cluster")             
        st.plotly_chart(fig, use_container_width = True)
        st.write("""This cluster has seen a massive spike in activity as social media posts about the fight in the Donbas,
        specifically around the city of Bakhmut. """)
        center_of_mass_plot(5)
        st.write("""This map shows the average location of the social media posts in this cluster.
        The location is nearby the city of Bakhmut, where fighting has been most intense as of April 2023.""")
    elif option == 'Satellite Imagery':
        st.header('''Satellite Imagery''')
        st.write("""Cluster 6: Satellite Imagery. This cluster is focused all over Ukraine and in neighboring countries, and is associated with images and reporting that uses satellite imagery. There have been noteworthy spikes in this cluster since the conflict began, and seems to be indicative of potential uncovering of human rights abuses and atrocities. For example, there was a spike of activity in April 2022, when satellite imagery was used to uncover human rights abuses in the city of Bucha outside Kyiv. This was again repeated in November of 2022, when Ukraine concluded its counteroffensive and took back territory in Kherson and Kharkiv, and again satellite imagery helped uncover evidence of atrocities in the towns formerly controlled by the Russian forces. There has been a lull in this cluster recently, but any increase in activity in this cluster might be indicative of using satellite imagery to document cases of activities that journalists or people keyed in to social media cannot get to.""")
        st.image('https://i.imgur.com/vv9QClc.png', caption = 'This is a generated word cloud image of the most-used words for this cluster, based on a training set of 10,000 posts')
        fig = go.Figure([go.Scatter(x=activity_counts_df['cal_date'], y=activity_counts_df['Cluster_6'])])
        fig.update_traces(mode="markers+lines", hovertemplate='Date: %{x} <br>Number of posts: %{y} <extra></extra>')
        clustered_bollinger_bands(6)
        fig.add_trace(go.Scatter(x = activity_counts_df['cal_date'], y = activity_counts_df['sma'] + (activity_counts_df['std'] * 2), line_color = 'gray', line = {'dash': 'dash'}, name = 'Upper Band', opacity = 0.3))
        fig.add_trace(go.Scatter(x = activity_counts_df['cal_date'], y = activity_counts_df['sma'] - activity_counts_df['sma'], name = 'Lower Band', opacity = 0.3, fill = 'tonexty', line_color = 'gray', line = {'dash': 'dash'}))
        fig.add_annotation(x=activity_counts_df.loc[activity_counts_df['cal_date'] == '5/05/2023'].to_json(), y=31, #I couldn't get annotations to work exactly how I wanted, so I'm pivoting
            text="Shaded areas identify number of social media posts in an expected 21-day average. <br> Anything not within this band signifies anomalous activity and may be indicative of a newsworthy development.",
            showarrow=True)
        fig.update_layout(showlegend=False, title = "Social media posts per day in this cluster")
        st.plotly_chart(fig, use_container_width = True)
        st.write("""This cluster's activity spikes when satellite imagery is used in social media posts.
        Notably, satellite imagery has been used to augment coverage of human rights abuses in Ukraine, and spikes in this cluster have focused on these posts.""")
        center_of_mass_plot(6)
        st.write("""This map shows the average location of the social media posts in this cluster.
        It is more centrally located, as many of the posts are referencing activities along the conflict's front and along the coast of Ukraine.""")            