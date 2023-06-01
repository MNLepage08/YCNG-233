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
  - Simplest model (Benchmark): Prediction = mean of the time series, Prediction = the last seen value.
  - 

#### 2. Time series classification
  - Supervised Learning: Dataset D = {X,Y} (X = [x1(t), ... ,xN(t)], Y = [y1, ... , yN] | yi is defined in a list of K classes) containing N time series and N labels. The task is to find for a new time series x(t) the corresponding class. Classification problem.
  - Unsupervised or semi-supervised: D = {X}, X = [x1(t), ... ,xN(t)]. The task is to detect if a new time series x(t) have a similar behavior than time series in D. Anomaly detection.
  - Ontology: Feature based (x(t) --> Feature extraction --> Classifier --> Class), Distance based (x(t) --> Distance --> Class), Deep Learning (x(t), Classifier --> Class)
  - Feature based methods: The main idea is to extract relevant information from the time series and provide it to a classification algorithm. Simple feature ex: mean, variance, RMS. Energy/power features: Shannon entropy, coefficient from DFT (Discrete Fourier Transformation). Correlation features: number/position of the peaks in the autocorrelation... Limits: Features must be defined, High dimensionality, Non stationarity, Time structure is not considered.  [TsFresh Library](https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html)
  - Distance based methods: Shapelet, SAX, BOSS, BOSSVS, HIVE-COTE,  DTW. Pattern based and tolerate some flexibility on signal amplitude. [Pyts Library](https://pyts.readthedocs.io/en/stable/), [DTW Library](https://pypi.org/project/dtw-python/)
  - [Assingment 2: ](https://github.com/MNLepage08/YCNG-233/blob/main/Time%20Series%20-%20Course%202.ipynb) Time Series with distance based methods (Shapelet & SAX-VSM)

#### 3. Time series forecasting
  - Concepts (Now, History, Step Size, Horizon): Now: the time where the prediction takes place. The “now” can be arbitrarily set. Each “now” will produce a new row. Step size: the time between 2 consecutives “now”s. Should be linked to the business problem. Do you need to do a prediction for each minute? Day? Month? History: For a given “now” how long in the past will you look at. Horizon: Number of steps in the future we would like to predict.
  - Preprocessing: Parse dates, Resample (sum / mean), Create X lags, Create Y outputs.
  - Evaluation metrucs: Mean absolute error (MAE), Root mean sqared error (RMSE), Mean absolute percentage error (MAPE), Symmetric mean absolute percentage error (sMAPE).
  - [Evaluation Strategy: ](https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/) k-fold Cross Validation Does Not Work For Time Series Data and Techniques That You Can Use Instead. Train-Test split that respect temporal order of observations. Multiple Train-Test splits that respect temporal order of observations. Walk-Forward Validation where a model may be updated each time step new data is received.
  - ARIMA Family model: AR(p), MA(q), ARMA(p, q), ARIMA(p, d, q), SARIMA(p, d, q)(P, D, Q, s) 
  - Exponential Smoothing: Suitable method when no clear trend or seasonality can be observed.

#### 4. Statistical methods for time series forecasting


6. Deep Learning for time series
7. Ensemble learning and its application on time series problems
8. Transfer learning and its application on time series problems
9. Graph neural networks and their applications on time series problems
10. Sequence models and their applications on time series problems
11. Project presentation

#### Bibliography
* [Forecasting: Principles and Practice](https://otexts.com/fpp3/)
