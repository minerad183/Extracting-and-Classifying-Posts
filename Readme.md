# Working Readme

# Extracting, Modeling, Clustering, and Displaying Geolocated Social Media Posts from the Russia-Ukraine War

**Author: Adam Miner**

## Problem Statement and Context

Russia's February 2022 invasion of Ukraine marked the end of a two-decade peace in Europe, and is the largest land war in Europe since World War II. This war is also one of the first instances of a war fought in the social media and information space, as well. With the number of smart phones and people connected to the internet, both in Ukraine and around the world, the Russia-Ukraine conflict has been cataloged like no other war before it. Now media organizations are not the only ones covering the war, everyday people can do it by just taking a picture or a video and posting it to social media sites like Twitter, or Telegram. Whole ecosystems have sprouted up to facilitate it, such as [individual users](https://twitter.com/RALee85?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor) on Twitter aggregating media posts, to loosely moderated threads on message boards like [Reddit](https://www.reddit.com/r/UkraineWarVideoReport/).

There are also non-government organizations trawling through social media and capturing as much information about the posts. Analysts employed by these organizations use their skills and expertise to construct narratives and fill in the gaps that the data from social media might have missed. In many cases, these organizations are also making their data freely available to other organizations and governments for their own use. Some examples of these organizations are [Bellingcat](https://ukraine.bellingcat.com/), [Texty.ua](https://texty.org.ua/projects/107577/under-attack-what-and-when-russia-shelled-ukraine/), [C4ADS](https://eyesonrussia.org/), and [@GeoConfirmed](https://geoconfirmed.azurewebsites.net/). These disparate organizations are using similar sources to analyze and display their data, but data may be missed by one or more. That is why it is imperative to utilize all that is out there to obtain a complete picture to track the conflict. 

Social media data represents a necessary component of Ukraine battle tracking. With all of these organizations doing the hard part of tracking and gathering the data, it falls on other groups to analyze the data in a meaningful way. There currently exist a few problems that this capstone project hopes to rectify:

1. The data are not made available in a one-stop-shop location, and require data science expertise to gather, extract-transfer-load, and store.
2. The way the data are collected and labeled is not standardized across different sources, and need to be standardized in a more digestable format.
3. The data and its contents can be analyzed more holistically, across space and time, and models built out of it.
4. The data should be displayed in an understandable, relevant, and useful format, such as through charts and maps, on a, interactive website dedicated to this.
5. The data should be updated on a regular (daily, hourly?) basis.

With these problems in mind, I will use the capstone project to address points one through four in some respects.


### Clean and standardize the data
Duplicate posts should be removed, and any artifacts or irrelevant or incorrect information should be removed. I also will need to look at how or if data are being classified and categorized, and see if there is a similar typology used across the datasets. If not, create my own typology and labeling structure, and add these labels to the data. This will be important for the modeling process.

### Create a Clustering Model to evaluate the records for action and actor
Using the typology created, design and implement a clustering model that takes in all the data and predicts what type of activity the social media post is cataloging, and who is the commitor of the activity (Russia, Ukraine, something else). The clustering model will use Natural Language Processing (NLP) to take the text information for each of the social media posts. It will also incorporate some sort of geographic component if applicable and useful, such as missile attacks striking central Ukraine categorized as a missile attack from Russia. Try any number of models with different parameters to attempt and create a highly accurate model on the already existing data, and then test it on the new data entering the database.

### Create a website that features the data, the model, and analysis of the data
Using the [Streamlit API](https://streamlit.io/) and hosted on [Render](https://extracting-and-clustering-posts-webapp.onrender.com/), create viewable charts and graphs, and a map, of the data with relevant analysis. The model's predictions should be used as data features in this website.


**Possible Questions to Answer**
1. Is it possible to ETL these datasets and store them in one concise, easy-to-understand data table? Or will work on a relational database need to be done?
2. Can I create a highly accurate clustering model that can identify unqiue clusters?
3. What kind of analysis can I discover from this? Do social media posts types change over time? Do certain post types and activity spike before or during intense periods of conflict? Are there any tippers or indicators to conflict increasing or decreasing?

## Analysis and Conclusion

### Workflow

1. Using the OSINT Geo Extractor package, gather data from different data sources. Scrape web map applications for labeling data and join the values together.
2. Process the data so it is ready for document vectorizing.
3. Run the Vectorizing model on the data, and run a clustering model on the data.
4. Analyze the generated clusters' textual, spatial, and temporal components.
5. Report findings via a web application using Streamlit.
6. Deploy the web application to the cloud on Render. Host data on AWS' S3.

### Data Gathered

Using the [OSINT-GEO-Extractor package](https://pypi.org/project/osint-geo-extractor/), gather the data, and synthesize and store it using an ETL pipeline. Columns that are similar between datasets should be combined, and extra columns referring to type of activity, who committed the activity, and other identifying information, should be created.

The OSINT-GEO-Extractor allows the data scientist to gather data from these organizations' databases:
- Bellingcat
- Ceninfores
- Defmon
- @Geoconfirmed
- Texty.ua

An important note is that the extractor misses some data, and scraping the websites themselves for this additional data (the labeling these groups assigned to these social media posts) resulted in more robust clustering modeling. See notebook 01_data_extraction.

### Pre-processing and Modeling

The data was prepared for document vectorization. Most of the preparations consisted of regular expressions and stemming to strip down the text info for more optimal clustering. A custom stopword list was also used, and augmented through multiple rounds of trial.

The stemmed data was then run through TF-IDF Vectorizer to generate numeric feature data for the next stage of modeling.

Through analyzing a set of K neighbor's silhouette scores, it was determined that seven clusters resulted in the most optimal set of explainability and uniqueness. Note that the silhouette scores indicated a mediocre fitting model. See notebook 02_unsupervised_clustering.

### Exploratory Data Analysis

There are a lot of interesting information that was cleaned from the clustering. See 03_clusters_spatial_temporal_eda notebook for the most information, or check out the web application for a more visually appealing description.

**Here is the summary of analysis of each of the clusters based on social media post text, spatial analysis, and temporal analysis:**

#### Summary of EDA and Identification of Clusters

- **Cluster 0: Russian Movements and Activities**. This cluster is focused on Russian movements, troops, and vehicles, inside Ukraine, Russia, and Belarus. This cluster is predictive in the long-term, as announcements or observations of movement are reported on early on, and again reported once the movement has taken place, typically at least a week out. This cluster is helpful to look at as a precursor to offenses, as spikes in activities are associated with Russian mobilizations and deployments. <br><br>

- **Cluster 1: Global Russian and Ukrainian Activities**. This cluster has the most amount of posts in it, has a much more global dispersal, and is more generalizable, focused on Russian and Ukrainian-related activities writ-large. This cluster's activity has stayed relatively consistent since November 2022, at a lower level than previously. It will be interesting if other clusters evolve to envelop more of the data as the conflict continues.<br><br>

- **Cluster 2: The Siege of Mariupol**. This cluster is directly related to activities around the siege of the city of Mariupol. It is geographically focused on the city, and the timeline of events back it up. The siege was initiated early on in the conflict, was reported on as the city was bombed and assaulted by Russian soldiers, and then eventually activity lulled when the Ukrainian soldiers surrendered. Recently activity spiked due to Russian President Vladimir Putin visiting the city in March 2023. There is predicted to be little activity in this cluster, though if activity in this cluster picks up it might be indicative of a Ukrainian push to retake the city.<br><br>

- **Cluster 3: The Destruction Cluster**. This cluster is focused on destruction wrought by both Russia and Ukraine, as the geographic locations, presumably locations of shelling and other attacks, of these activities are contained largely within Ukraine and Russia. This cluster seems to be related to offenses taken by either side, as well as lulls in fighting as artillery and materiel supplies dwindle. There has recently been an increase in activity in this cluster, as Russia had initiated an offensive in the Donbas region of Ukraine.<br><br>

- **Cluster 4: Ukrainian Positions and Activities**. This cluster is focused on what Ukraine is doing in its battle plans. Activities related to troop movements, positioning, and UAV flights. Geographically the locations of the activities are focused in Eastern Ukraine, as there is a focus on defending the Donbas region. Previous spikes in activity have occurred when Ukraine was moving troops to retake parts of the country. Notably, this cluster has seen a significant decrease in activity recently, but very likely any increase in activity in this cluster means a possible Ukrainian counteroffensive.<br><br>

- **Cluster 5: Battle for Bakhmut**. This cluster is focused on the Donbas, specifically the Donetsk Oblast in Ukraine. It is primarily concerned with activities around the besieged city, which has seen intense fighting since the late Fall, as Russia has focused its efforts on taking this city. It has very recently seen a massive spike in activity, as Russia's offsensive and Ukraine's fierce defense has resulted in the most active part of the conflict right now. Continued increase in this cluster is going to be indicative of intense fighting for the city. If this cluster decreases, it likely means the city has been taken over or is on the verge of being taken over by either side.<br><br>

- **Cluster 6: Satellite Imagery**. This cluster is focused all over Ukraine and in neighboring countries, and is associated with images and reporting that uses satellite imagery. There have been noteworthy spikes in this cluster since the conflict began, and seems to be indicative of potential uncovering of human rights abuses and atrocities. For example, there was a spike of activity in April 2022, when satellite imagery was used to uncover human rights abuses in the city of Bucha outside Kyiv. This was again repeated in November of 2022, when Ukraine concluded its counteroffensive and took back territory in Kherson and Kharkiv, and again satellite imagery helped uncover evidence of atrocities in the towns formerly controlled by the Russian forces. There has been a lull in this cluster recently, but any increase in activity in this cluster might be indicative of using satellite imagery to document cases of activities that journalists or people keyed in to social media cannot get to.

#### Web Application Design

There are two web application python files in this repo. The first one, in the displays folder, contains the complete web application using the Streamlit API. This application allows users to view an interactive map of the clustered data, the results of the EDA notebook filtered by each cluster, plus descriptions and limitations of the model in a visually appealing way. Notably, this application also allows users to add their own text in a text block, and the model will analyze the text contents and generate a prediction for that text.

The second web application python file, called application and in the main folder, is the application file that creates the cloud-based web app hosted on Render. Unfortunately, there were many issues attempting to put the prediction algorithm into this cloud-based web application that were outside of my control, so this web application shows everything but that. If a software developer with more experience accessing NLTK's data (or other SDK data) would like to bring the prediction model back in, that would be greatly appreciated.

#### Classification Model

The last stage in this project was to create a classification model off of the clustered data. This classification model is what runs on the user-inputted text information.

Doing a classification prediction model on the user input, versus doing another K-means clustering model is more optimal because the K-means clustering model needs more inputs and is a little "fuzzier" in how it assigns the clusters. Since it is unsupervised learning, I modeled the large set of training data in order to generate the Y values. Now that I have the Y values, I can use a classification approach to predict inputs based on them. This approach will discount any changes in clusters that are generated, however, so periodic recycling of the clusters and a retraining classification model should be in order.

## Recommendations for future and further analysis
1. Expand the dataset to include other sources of filtered conflict data. There are several of these groups that do similar collection like GeoConfirmed. Combining the data together (as well as removing duplicate social media posts) will make for a stronger dataset to train models on.
2. Refine the process of cleaning, stemming, and vectorizing the data. For example, instead of TF-IDF algorithm, use <a href="https://towardsdatascience.com/multi-class-text-classification-with-doc2vec-logistic-regression-9da9947b43f4" title="Doc2Vec">Doc2Vec.</a>
3. Utilize a different clustering algorithm, like <a href= "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html"title="DBSCAN">DBSCAN.</a> This will hopefully allow for a stronger, more fitting, more cluster impactful model.
4. Incorporate more spatial analysis into the model. For example, every post in these datasets have geolocated post data. Use spatial point pattern analysis to determine similar posts not only in text information, but location-based information.
5. Rerun the model on a month's worth of recent data. Data were collected up until April 20th, 2023. Test the model on data from this point on to see how the model fits new events, and whether assertions and judgements made in the cluster analysis hold true.

# Appendix

## Data Dictionary

### Repo Folders
| Folder Name | Description |
| ------ | ----------- |
| data   | path to data files, pickled files, and returned csvs. |
| displays | folder that contains screenshots and images used in the web app and presentation. Also includes an html interactive map, and the Streamlit API completed web  application  code|
| notebooks    | location of Jupyter Notebooks that the models were created on. |
| presentation    | location of the presentation files. |

### Repo Files
| File Name | Description | Folder Location |
| ------ | ----------- | ----------- |
| activity_counts.csv   | Created CSV used for time series modeling |data |
| bellingcat.csv   | Data from Bellingcat using the OSINT Geo Extractor and web scraping |data |
| ceninfores.csv   | Data from Ceninfores using the OSINT Geo Extractor|datasets |
| class_model.pkl   | Pickle file of the classification model for the prediction algorithm|data |
| defmon.csv   | Data from Defmon using the OSINT Geo Extractor |data |
| geojson.json   | GeoJSON information of the Geoconfirmed dataset |data |
| get_geoconfirmed_data.csv   | Processed Geoconfirmed data, not yet clustered |data |
| get_geoconfirmed_data_clusters.csv   | Processed and clustered Geoconfirmed data  |data |
| km_model.pkl   | Pickle file of K-means clusterint model |data |
| stopwords.zip   | NLTK's stopwords zipped file, used for the text processing |data |
| texty.csv   |  Data from Texty.ua using the OSINT Geo Extractor |data |
| vectorizer.pkl   | Pickle file of TF-IDF Vectorizer  |data |
| application_streamlit_local.py   | Completed Streamlit web application, but only deployable to a local machine in its current state |displays |
| map.html   | HTML interactive map |displays |
| clust0_movements_wc_f.jpg   |  Russian Movements cluster word cloud image |displays |
| clust1_globalactivities_wc_f.jpg   |  Global Activities cluster word cloud image |displays |
| clust2_mariupol_wc_f.jpg   |  Siege of Mariupol cluster word cloud image |displays |
| clust3_destruction_wc_f.jpg   |  Destruction cluster word cloud image |displays |
| clust4_ukrposact_wc_f.jpg   |  Ukrainian Activities cluster word cloud image |displays |
| clust5_battlebakhmut_wc_f.jpg   |  Battle for Bakhmut cluster word cloud image |displays |
| clust6_satelliteimagery_wc_f.jpg   |  Satellite Imagery cluster word cloud image |displays |
| application   | Application hosted on Render  |main |
| 01_data_extraction.ipynb   | Notebook to pull data using web scraping and the OSINT Geo Extractor package  |notebooks |
| 02_unsupervised_clustering.ipynb   | Notebook to do the preprocessing and K-means clustering  |notebooks |
| 03_clusters_spatial_temporal_eda.ipynb   | Text, spatial, and temporal EDA notebook  |notebooks |
| 04_classification_model.ipynb   | Classification model on the clustered data, for predictions |notebooks |
| Miner_Presentation_Project_3.pptx   | Powerpoint presentation slides  |presentation |
| miner_nlp_acolyte_dnd_classifiers.pdf   | PDF version of powerpoint presentation slides  |presentation |

### Relevant and Featured Data and Columns

| Data/Columns | Description | Original Data if applicable |
| ------ | ----------- | ----------- |
| list_text   | Full text information for the social media posts, for ease of clustering  |Title + Description|
| Cluster_0 through _6   | Activity of social media posts per day for each cluster. These columns in the activity_data.csv helped do time series analysis and create the interactive line chart | Date and Clusters |
| Clusters   | Column assigning each post one of the seven unique clusters | list_text|
| Ukraine_bnd   | A geojson simplified boundary file for Ukraine. Used to do a spatial join on data from each cluster to generate the proportion of posts within the country |N/A  |

## Sources
- all our lecture repositories on Github
- Streamlit
- Render
- Amazon Web Services S3
- Mapbox web service
- https://ukraine.bellingcat.com/
- https://texty.org.ua/projects/107577/under-attack-what-and-when-russia-shelled-ukraine/
- https://eyesonrussia.org/
- https://geoconfirmed.azurewebsites.net/
- https://www.scribblemaps.com/maps/view/2022051301800/nBT8ffpeGH
- https://pypi.org/project/osint-geo-extractor/
- https://www.tutorialspoint.com/gensim/gensim_doc2vec_model.htm#:~:text=Doc2Vec%20model%2C%20as%20opposite%20to,the%20words%20in%20the%20sentence.
- https://www.kaggle.com/code/currie32/predicting-similarity-tfidfvectorizer-doc2vec/notebook
- https://www.datacamp.com/tutorial/wordcloud-python
- https://github.com/wmgeolab/geoBoundaries/
- https://machinelearningmastery.com/multinomial-logistic-regression-with-python/
- https://albertum.medium.com/plotting-bollinger-bands-with-plotly-graph-objects-1c7172899542
- https://spatialreference.org/ref/epsg/20007/
- https://towardsdatascience.com/how-to-create-interactive-map-plots-with-plotly-7b57e889239a
- https://towardsdatascience.com/build-a-multi-layer-map-using-streamlit-2b4d44eb28f3
