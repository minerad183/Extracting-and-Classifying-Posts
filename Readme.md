# Working Readme

# Extracting, Modeling, Classifying, and Displaying Geolocated Social Media Posts from the Russia-Ukraine War

**Author: Adam Miner**

## Problem Statement and Context

Russia's February 2022 invasion of Ukraine marked the end of a two-decade peace in Europe, and is the largest land war in Europe since World War II. This war is also one of the first instances of a war fought in the social media and information space, as well. With the number of smart phones and people connected to the internet, both in Ukraine and around the world, the Russia-Ukraine conflict has been cataloged like no other war before it. Now media organizations are not the only ones covering the war, everyday people can do it by just taking a picture or a video and posting it to social media sites like Twitter, or Telegram. Whole ecosystems have sprouted up to facilitate it, such as [individual users](https://twitter.com/RALee85?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor) on Twitter aggregating media posts, to loosely moderated threads on message boards like [Reddit](https://www.reddit.com/r/UkraineWarVideoReport/).

There are also non-government organizations trawling through social media and capturing as much information about the posts. Analysts employed by these organizations use their skills and expertise to construct narratives and fill in the gaps that the data from social media might have missed. In many cases, these organizations are also making their data freely available to other organizations and governments for their own use. Some examples of these organizations are [Bellingcat](https://ukraine.bellingcat.com/), [Texty.ua](https://texty.org.ua/projects/107577/under-attack-what-and-when-russia-shelled-ukraine/), [C4ADS](https://eyesonrussia.org/), and [@GeoConfirmed](https://geoconfirmed.azurewebsites.net/). These disparate organizations are using similar sources to analyze and display their data, but data may be missed by one or more. That is why it is imperative to utilize all that is out there to obtain a complete picture to track the conflict. 

Social media data represents a necessary component of Ukraine battle tracking. With all of these organizations doing the hard part of tracking and gathering the data, it falls on other groups to analyze the data in a meaningful way. There currently exist a few problems that this capstone project hopes to rectify:

1. The data are not made available in a one-stop-shop location, and require data science expertise to gather, extract-transfer-load, and store.
2. The way the data are collected and labeled is not standardized across different sources, and need to be standardized in a more digestable format.
3. The data and its contents can be analyzed more holistically, across space and time, and models built out of it.
4. The data should be displayed in an understandable, relevant, and useful format, such as through charts and maps, on a, interactive website dedicated to this.
5. The data should be updated on a regular (daily, hourly?) basis (_note, might be outside the scope of this project_).

With these problems in mind, I will use the capstone project to address all of these.

### Making data available
Using the [OSINT-GEO-Extractor package](https://pypi.org/project/osint-geo-extractor/), gather the data, and synthesize and store it using an ETL pipeline. Columns that are similar between datasets should be combined, and extra columns referring to type of activity, who committed the activity, and other identifying information, should be created.

The OSINT-GEO-Extractor allows the data scientist to gather data from these organizations' databases:
- Bellingcat
- Ceninfores
- Defmon
- @Geoconfirmed
- Texty.ua

### Clean and standardize the data
Duplicate posts should be removed, and any artifacts or irrelevant or incorrect information should be removed. I also will need to look at how or if data are being classified and categorized, and see if there is a similar typology used across the datasets. If not, create my own typology and labeling structure, and add these labels to the data. This will be important for the modeling process.

### Create a Classification Model to evaluate the records for action and actor
Using the typology created, design and implement a classification model that takes in all the data and predicts what type of activity the social media post is cataloging, and who is the commitor of the activity (Russia, Ukraine, something else). The classification model will use Natural Language Processing (NLP) to take the text information. It will also incorporate some sort of geographic component if applicable and useful, such as missile attacks striking central Ukraine categorized as a missile attack from Russia. Try any number of models with different parameters to attempt and create a highly accurate model on the already existing data, and then test it on the new data entering the database.

### Create a website that features the data, the model, and analysis of the data
Using the [Streamlit API](https://streamlit.io/), or some other web service, create viewable charts and graphs, and a map, of the data with relevant analysis. The model's predictions should be used as data features in this website.

### Gather data on a timely basis (might be outside scope)
The databases from these services update on a schedule. Find a way to process and ETL the data, remove old entries from the synthesized data table in the database, and rerun the model on these new data. This will generate the predictions from the model (which should then be evaluated to make sure the model is valid for new data), and publish the new data to the created website.

**Possible Questions to Answer**
1. Is it possible to ETL these datasets and store them in one concise, easy-to-understand data table? Or will work on a relational database need to be done?
2. Can I create a highly accurate classification model that can correctly predict a multiclass dependent variable? We'll shoot for above 90 percent accuracy on a classification model, in order to limit how much time people have to go back into the dataset and reclassify some of the entries that were mislabeled.
3. What kind of analysis can I discover from this? Do social media posts types change over time? Do certain post types and activity spike before or during intense periods of conflict? Are there any tippers or indicators to conflict increasing or decreasing?
