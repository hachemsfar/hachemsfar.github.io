# hachemsfar.github.io

# Drought Level Prediction in the U.S.

*This project was carried out as part of the TechLabs “Digital Shaper Program” in Aachen (Summer Term 2022)*

## 1\. Introduction

As autumn approaches, the west drift re-establishes itself as the dominant driving force behind the typical European weather, engulfing the continent in cool and moist cyclonic air masses from the Northern Atlantic. With them, comes the long anticipated rain to alleviate the severe state of drought that has been affecting most countries in the region.

The year 2022 was atypical, but also lines up with the concerning trend of increasingly hotter and drier weather that has been observed in Europe in the past years. With an exceptional high pressure system taking the lead in early March, the atmospheric circulation above the continent remained almost permanently disturbed until the month of August, making way for subtropical dry air instead of the cooler Atlantic air. The consequence is a significant decline in precipitation, inclement heat waves and the onset of a drought period, referred to by the Global Drought Observatory as possibly the worst since at least 500 years. By late August, two-thirds of Europe were under a drought warning, drawing attention to stressed water levels and crippled crops. The impact extends from economic to environmental concerns and even life-threatening shortages: harvests are predicted to take a hit; the cost of freight already increased as major rivers dry up and limit the capacity of vessels on shallow waterways; emissions reached a new high as wildfires raged across the continent; several power plants have been offline for weeks due to the lack of water flow to cool down the reactors or spin the turbines and in France alone, more than 100 municipalities have declared state of emergency, as drinking water supply could no longer be guaranteed. The following figure shows the extent of the described drought situation.

<center><img src="../_resources/Combined Drought Indicator.jpg" width="500" class="jop-noMdConv"></center><center>Figure 1: Combined Drought Indicator (CDI 2.1) for Europe in early August 2022 (Source: Global Drought Observatory)</center>

The problem of increasingly frequent droughts as a result of climate catastrophe is not a European issue. Severe droughts have stricken other parts of the world in 2022, with similar consequences: In the United States, Lake Mead and Lake Powell on the Colorado River have reached historically low levels, raising concerns about the livelihood of 40 million people who rely on the system for freshwater. Figure 2 below shows the comparison between 2017 and 2022. In China, the Yangtze River – the third longest river in the world – is withering away and putting 450 million people along its basin on alert. In the Horn of Africa, yet another long lasting drought in an otherwise already economically depressed region puts over 20 million people, according to a report of the United Nation’s World Food Programme, at risk of starvation in the next months, should the situation not improve.

<center><img src="../_resources/Portion of Lake Powell in Utah.jpg" width="500" class="jop-noMdConv"></center><center>Figure 2: Portion of Lake Powell in Utah, in the summer 2017 and summer 2022 (Source: NASA)</center>

As the climate catastrophe progresses, persistent deviations from the typical weather patterns are expected to become more common. The decreasing temperature difference between the Earth’s poles and the equator takes away the system’s dynamic nature and causes the Jetstream to slow down. With less momentum, this band of high velocity winds near the tropopause starts to meander, creating a barrier that stagnates the local circulation and leads to very stable, long lasting weather patterns.

Beyond the prediction of rainfall as a task covered by meteorology, inferring the effect of said rainfall on the soil and hydrology is a challenge on its own. Ideally, soil moisture should be monitored by probes buried at varying depths, but as of today no such system has seem widespread deployment. As a result, physical and statistical models are instead used to process meteorological data gathered by regular weather stations and produce a result that reflects the state of vegetation as well as ground water levels. Such models are pieced together by national weather services and international observatories, with the intent of helping both governmental bodies and individuals to identify a drought and prepare accordingly for possible consequences. Recognizing a state of drought isn’t as trivial as it may appear, since the soil’s stratification or composition coupled with underground reservoirs cannot be easily assessed by a casual observer at surface level. Furthermore, soil moisture levels are taken into consideration by global weather forecast systems, such as the ECMWF, DWD’s ICON or NWS’ GFS. Because the moisture contained in the soil must be in equilibrium with the atmosphere, and water vapor directly impacts the heat capacity of air, weather forecasts heavily rely on it. An inadequate estimate of this key parameter leads to uncertain forecasts and eventually to the inability to predict wildfires or the intensity of heat waves and their associated wet bulb temperature, which explores the effect of temperature and humidity on the human body. This particular phenomenon could be observed in Central Europe during the summer of 2022. Due to underestimated evapotranspiration, the GFS model was consistently off target by a few Kelvin regarding peak temperatures at noon.

With the aforementioned considerations, it becomes clear that drought prediction models constitute a vital part of our efforts to understand the world we live in. They are also necessary to ensure stable living conditions in times of drastic changes. The goal of this project is to develop a simplified version of such a model with tools provided by data science and the Python programming language. By applying different machine learning algorithms, we will try to predict drought periods in the form of a drought level score. The project has originally appeared as a Kaggle coding challenge [Kaggle Drought-Prediction Dataset](https://www.kaggle.com/datasets/cdminix/us-drought-meteorological-data) and included already split datasets from the United States.

## 2\. Methodology

### 2.1 Getting a first glimpse of the data at hand

In order to determine the actual scope of the project and set achievable goals, it was first necessary to understand the data we would be dealing with. Provided were several datasets for a total of more than 22 million rows: a training, validation and testing dataset with daily meteorological records of over three thousand counties in the U.S., covering a period of twenty years. Additionally a dataset with static soil parameters for each county was given.

The soil dataset contained detailed geographical information, such as location, elevation, land cultivation, presence and extension of waterbodies, as well as terrain slope or direction. As opposed to the main meteorological datasets, many of these features were static and did not vary over time for each county. There was uncertainty in the group about how this static data should flow into our later machine learning models. While interactions with the main datasets were assumed to exist to a certain point, incomplete data from this set could introduce noise to the model and eventually jeopardize the entirety of the project. Due to that as well as time and resource constraints, it was decided in plenary not to include this data. Hence, only the meteorological data would be included in our machine learning models and the soil dataset was not treated any further. This decision was made with the knowledge of possibly losing predictive accuracy.

The following figure displays the given parameters of the meteorological datasets. Every parameter corresponds to a column in the datasets.

<center><img src="../_resources/Variablenlist.jpg" width="500" class="jop-noMdConv"></center><center>Figure 3: List of given parameters</center>

All parameters, including the score and excluding the date and fips, were continuous. That the score was continuous was surprising, since the description of the datasets indicated that the score corresponded to the drought categories of the U.S. Drought Monitors, as shown in the figure below.

<center><img src="../_resources/Classifaication of drought levels by the US.jpg" width="300" class="jop-noMdConv"></center><center>Figure 4: Classification of drought levels by the U.S. Drought Monitor \\\[1\\\]</center>

It became clear that the original U.S. Drought Monitor dataset had been modified. This sent the team off on a hunt for the original data. We managed to track down the dataset’s author and reverse engineer the manipulation steps taken, revealing that the categorical score was changed to a continuous score ranging from 0 to 5 by weighting it with the relative area affeced by each drought level . Hence, a score of 1 corresponds to D0, a score of 2 to D1, and so on.

We also decided to treat only one of the approximately 3100 counties. The first county in the dataset with the fips code 01001 was selected, corresponding to Autauga in the state of Alabama.

### 2.2 Data preparation and processing

At the first glimpse of the given datasets, it was also identified that the drought score is only given weekly as shown below. All other meteorological data were presented daily.

<center><img src="../_resources/First 100 days.jpg" width="500" class="jop-noMdConv"></center><center>Figure 6: Drought score variable of the first 100 days</center>

We decided to deal with this problem in two different ways and to create two types of datasets. The first type of dataset was generated by interpolating the missing score entries. The second type of dataset was created by aggregating the given values on a weekly basis, similar to how the score was originally computed by the U.S. Drought Monitor. The weekly mean values of all meteorological variables were selected for this purpose. The creation of both types of datasets was chosen in order to later investigate whether this could influence the prediction quality. Those types are referred to below as the mean and interpolation datasets, respectively. As an example, the interpolated data set is shown below.

<center><img src="../_resources/after interpolation.png" width="500" class="jop-noMdConv"></center><center>Figure 7: Drought score variable of the first 100 days after interpolation</center>

### 2.3 Exploratory analysis for feature selection

For further and more in-depth examination of the data sets, we resorted to auxiliary tools such as histograms, boxplots and correlation matrices. All these tools were applied to the untouched original data set, i.e. before interpolating and taking the mean. From the histograms and boxplots of the meteorological parameters we could not draw any valuable conclusions, so in the following only the correlation matrix is listed.

<center><img src="../_resources/Correlation matrix of all variables.jpg" width="700" class="jop-noMdConv"></center><center>Figure 7: Correlation matrix of all variables used</center>

The matrix displays the correlations between each variable and, more importantly, how the drought score relates to every one of them. A few obvious relations can directly be observed. All wind speed variables are negatively correlated to the temperature above ground. Another observation is that surface pressure is negatively correlated to the total precipitation. However, when it comes to the drought score, no strong correlation can be found. The highest correlation with 0.12 - 0.18 exists with the variables of the temperature two meters above ground (T2M). Presumably, the reason there are no strong correlations is that the drought score depends on many factors simultaneously and none of them plays a dominant role. Hence, we decided to include all given meteorological quantities in our machine learning models.

### 2.4 Model selection and goal specification

After feature selection, data analysis and preparation were completed, we were able to apply the first machine learning algorithms. With dwindling capacity due to team members leaving the project to pursue other life goals, we decided to focus our efforts on two algorithms. We used the *Random Forest Regressor* and *Facebook's Prophet*. While the Random Forest Regressor is a classical algorithm of supervised machine learning, Prophet is an algorithm specifically designed for time series forecasting.

The Random Forest algorithm exists as a regressor and a classifier. We used the regressor because our target variable (the drought score) is continuous. The Random Forest algorithm was chosen due to it being quite beginner friendly in general. In addition, outliers have little impact on the results and linear, as well as non-linear relationships are handled well. Prophet on the other hand is a widely used time series forecasting algorithm and can be easily deployed.

Both algorithms were trained and tested on the same data, the beforehand created mean and interpolation datasets. The training data covers the period from 2000 to 2016. The validation data covers the years 2017 to 2018 and was used as a baseline to compare the predictions against. The testing data, ranging from 2019 to 2020 was not used in any way.

## 3\. Project Results

**Random Forest Regressor**
For the random forest regressor the Mean Absolute Error (MAE) was used as the diagnostic metric. For the models trained both on the mean and interpolation data, the MAE is 0.75. Considering that the drought level can take values from 0 to 5, 0.75 is a rather high variation.

The following two diagrams each show the comparison between predicted drought level and that actually observed during the period. The upper prediction was made with the mean data model and the lower one with the model based on interpolation data. Two things stand out. As the MAE has already indicated, prediction and actual values seem to differ greatly. It can also be seen that despite the different initial approach, the prediction course is very similar in both diagrams. In one case only more data points were output, which is why the curve looks more jagged. We conclude that it makes little difference whether the mean or interpolation data is used.

<center><img src="../_resources/Random Forest MEAN.jpg" width="600" class="jop-noMdConv"></center><center><img src="../_resources/Random Forest INTERPOLATION.png" width="600" class="jop-noMdConv"></center><center>Figure 8: Drought score predictions of Random Forest Regressor compared to actual observed validation data</center>

After its execution, the Random Forest algorithm allows to display the Feature Importance or, simply put, the impact of each singular parameter on the overall model output. This is displayed using the mean decrease in impurity (MDI). The following diagram shows that the maximum temperature 2 meters above ground (T2M_MAX) seems to have the greatest influence, distantly followed by the temperature range 2 meters above ground.

<center><img src="../_resources/Feature importance INTERPOLATION.jpg" width="600" class="jop-noMdConv"></center><center>Figure 9: Feature importance using MDI for interpolation model</center>

**Prophet**
Based on the understanding that the type of dataset used - mean or interpolation - has little influence on the results, we selected one for further procedure. This was the interpolated dataset. After Prophet was fed with it, the following diagram was construced.

<center><img src="../_resources/Comparison between actual values and the predictio.png" width="600" class="jop-noMdConv"></center><center>Figure 10: Comparison between Facebook's Prophet predictions (blue line) and the actual measured values (dotted line) of the whole period from 2000 to the end of 2018.</center>

Over the training period from 2000 to the end of 2016, it can be seen that Prophet is able to follow the fluctuations of the drought score quite well. From the beginning of 2017 until the end of 2018 the validation period begins (no more observed data points) and it can be seen how the algorithm immediately continues aligned with previous cycles. We compared this future prediction with the actual drought score from 2017 to 2018. The comparison is plotted on the diagram below. As with the Random Forest algorithm, appreciable deviations from reality can be observed.

<center><img src="../_resources/Comparison between validation data and the extende.png" width="300" class="jop-noMdConv"></center><center>Figure 11: Comparison between Facebook's Prophet predictions and the actual data points of the validation period</center>

As a last resort to improve the models's accuracy, we decided to try and craft a model blend, where the results of our machine learning algorithms would be combined in a linear superposition with an existing drought index known to be included by the U.S. Drought Monitor. For this purpose, the Standardized Precipitation Index (SPI) was chosen. The SPI constitutes a stochastic model recommended by the World Meteorological Organization as the main index for countries to follow drought conditions. Other indexes were either too taxing to run on consumer grade hardware, too complex to be implemented or just lacked any sort of literature and documentation. Unfortunately, the results were disappointing, as the SPI showed less variability than the score values and could not contribute to reducing the model's mean error. Figure 12 exemplifies this for a selected time window. Despite it being a key parameter in the calculation of the score, it is clearly far from the only one. This approach proved thus to be a dead end.

<center><img src="../_resources/spi-score-comparison.JPG" width="500" class="jop-noMdConv"></center><center>Figure 12: Comparison between the SPI score and the given score values for the same time period</center>

## 4\. Conclusion

The algorithms we applied could not make robust drought predictions. Whether this is because of the algorithms nature or because not enough influencing factors were included could not determined. Our hypothesis is that the poor prediction performance mostly results from not including the soil and geographic data, as well as lacking a longer training timeseries to feed the algorithms with. This could be a main aspect of future work, in addition to applying further and possibly more sophisticated machine learning algorithms. Classification rather than regression could also be tried. This would require either using the original U.S. Drought Monitor dataset, or converting back the existing continuous score into categories. It should also be remembered that only a single county could be analyzed. It is assumed that a more complex model including many other counties would significantly improve results.

Finally, we would like to briefly discuss to what extent prefabricated machine learning models, such as the Random Forest and others, are actually capable of predicting a complex phenomenon like the drought score. Certainly, machine learning applications can contribute enormously to weather and climate related research, but we believe that such complex issues as droughts can only be reliably predicted by multi-layered physical models that include all relevant quantities and are based on a more sound understanding of the underlying interactions. The models used in climate research are a very good example of this.

## Team

Tabib Ibne Mazhar - Mentor
Nusrat Jahan Suha - Mentor
Gabriel Fabrini Ribeiro - Group member
Jan Krusenbaum - Group member

## References

\[1\] [U.S. Drought Monitor: What is the USDM](https://droughtmonitor.unl.edu/About/WhatistheUSDM.aspx)
