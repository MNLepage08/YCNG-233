# YCNG-233
## Time Series Analysis Fundamentals

#### 1. Introduction to Time Series problems
  - Time series: a suite of data point ordered over time. The minimal information in a time series is the datetime (or the timestamp) of the datapoint and the value itself. The value can be discrete or continuous.
  - UTC is a continuous value, but it might be hard to extract information such as the season, the night/day... For data science purpose, the preferred representation is a datetime + explicit time zone. [Datetime Library](https://docs.python.org/3/library/datetime.html) & [Pytz Library](https://pypi.org/project/pytz/)
  - [Time zones and offset: ](https://youtu.be/-5wpm-gesOY)The Problem with Time & Timezones
  - [Resample:](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html) You cannot assume data are equally distributer overtime. You must check it. Resampling will help to Handle duplicates and Highlight missing data.
  - Filling missing data: Many ways. The 2 mostly used are [Interpolation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html) and [Forward Fill](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html)
  - Just as correlation measures the extent of a linear relationship between two variables, autocorrelation measures the linear relationship between lagged values of a time series. The [autocorrelation function](https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.acf.html) describes the relationship between a time series and its lagged counterpart. The [partial autocorrelation](https://www.statsmodels.org/devel/generated/statsmodels.tsa.stattools.pacf.html) describes a direct relationship, that is, it removes the effects of the intermediate lagged values.
  - [Time Series Decomposition: ](https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/) Involves thinking of a series as a combination of level, trend, seasonality, and noise components.
  - [Stationary vs. Non-Stationary: ](https://machinelearningmastery.com/time-series-data-stationary-python/) A stationary series is one where the values of the series is not a function of time. Mean(ts) = Mean(slide), Variance(ts) = Variance(slide), Autocorrelation(ts) = Autocorrelation(slice). Test: Augmented Dickey–Fuller test
  - [Assignment 1: ](https://github.com/MNLepage08/YCNG-233/blob/main/Time%20Series%20-%20Course%201.ipynb)Data Exploration on Covid-19 (Preprocessing, Correlation, ACF & PACF, Decomposition, Stationarity)

#### 2. Time series classification

4. Time series forecasting
5. Statistical methods for time series forecasting
6. Deep Learning for time series
7. Ensemble learning and its application on time series problems
8. Transfer learning and its application on time series problems
9. Graph neural networks and their applications on time series problems
10. Sequence models and their applications on time series problems
11. Project presentation

#### Bibliography
* [Forecasting: Principles and Practice](https://otexts.com/fpp3/)
