## Projecting the Future of Antarctica Sea Ice Using Reconstructed Data and ARIMA

I applied machine learning techniques to investigate the extent of Antarctica's sea ice for the next five years.

***

## Introduction 

Antarctica sea ice is characterized by unpredictability. Unlike Arctic sea ice, Antarctic sea ice has not followed a trend that most scientists and the public would expect. While Arctic sea ice has decreased steadily due to increasing global temperatures, the opposite was occurring in the south. Since the satellite era, Antarctica's sea ice has steadily increased until the 2015/2016 season, when the sea ice extent reached what was then the record low. Just before that season, Antarctica's sea ice reached its maximum extent ever recorded in 2014, further suggesting the unpredictability of sea ice. Ever since the decrease in the 2015/2016 season, Antarctica's sea ice has continued to decrease on average. Antarctica follows a seasonal pattern with September being the maximum and February being the minimum. In September of 2023, Antarctica's sea ice reached the lowest maximum ever recorded. While rising temperatures play a role, Antarctic sea ice is complicated in nature. Due to the complicated interactions between atmospheric and oceanic phenomena, locally and globally, the past and future of Antarctica's sea ice is difficult to predict. This was the case until recently published reconstructed Antarctic sea ice data. 

Recently, reconstructed Antarctic sea ice data by Fogt et al. has accurately modeled the increase and decrease recorded in the observation data, which is a problem continually missed by climate models. Using this reconstruction data set, I followed a machine learning approach to model the future of the Antarctic sea ice extent. I used an ARIMA (autoregressive integrated moving average model approach) to solve the problem. I conclude that the Antarctic sea ice extent for the next five years will continue to follow the current decreasing trend recorded in the past few years.


## Data

The primary data set for this project is from NSIDC Seasonal Antarctic Sea Ice Extent Reconstructions developed by Ryan Fogt et al. The data is reconstructed from 1905 to 2020 using temperature and pressure observation indexes. The best-fit model for reconstructed data is available for the entire Antarctic sea ice extent and the five geographic regions (Amundsen-Bellingshausen Seas, Weddell Sea, King Hakon VII, East Antarctica, and Ross-Amundsen Sea); however, the entire Antarctica is focused on this project from 1979 to 2020. 

A new dataset is combined to include all the seasonal sea ice extent for each year as this was seasonal reconstruction data. As a result, a new file from 1979 to 2020 is created with four points with data for each season. Figure 1 shows the plotted seasonally reconstructed data from  DJF (December, January, February), MAM (March, April, May), JJA (June, July, August), and SON (September, October, November). Autoregression would not be accurate and reliable with only four data points representing each year. The data was interpolated with 500 points to provide more data for the autoregression (Figure 2). Additionally, a new data set is created with 411 points for reconstructed sea ice extent and years from 1979 to 2020 in intervals of 0.1 years.  


![](assets/IMG/Seasonal Reconstruction Data.png){: width="500" }

*Figure 1: This is the best-fit for four seasons reconstructed from 1979-2020.*


![](assets/IMG/Interpolated Data.png){: width="500" }

*Figure 2: The interpolated data of sea ice extent using 500 points.*
## Modelling

Here are some more details about the machine learning approach, and why this was deemed appropriate for the dataset. 

The model might involve optimizing some quantity. You can include snippets of code if it is helpful to explain things.

```python
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_features=4, random_state=0)
clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
clf.fit(X, y)
clf.predict([[0, 0, 0, 0]])
```

This is how the method was developed.

## Results

Figure X shows... [description of Figure X].

## Discussion

From Figure X, one can see that... [interpretation of Figure X].

## Conclusion

Here is a brief summary. From this work, the following conclusions can be made:
* first conclusion
* second conclusion

Here is how this work could be developed further in a future project.

## References
[1] DALL-E 3

[back](./)

