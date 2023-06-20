# YCNG-233: Time Series Analysis Fundamentals

<p align="center">
<img width="500" alt="Capture d’écran, le 2023-06-04 à 09 54 32" src="https://github.com/MNLepage08/MNLepage08/assets/113123425/5145feb1-e4f7-4cd5-a5f6-bb684c78b947">

  
## :rocket: Assignments

1. [Data Exploration on Covid-19 for the new cases & deaths: ](https://github.com/MNLepage08/YCNG-233/blob/main/Time%20Series%20-%20Course%201.ipynb)Collect Covid-19 from Canada dataset. Preprocess (missing - duplicate - negative values, date conversion), Pearson correlation on new cases/deaths, ACF & PACF, decomposition and stationarity of time series.<p>

2. [Time Series classification with distance based: ](https://github.com/MNLepage08/YCNG-233/blob/main/Time%20Series%20-%20Course%202.ipynb)Collect Italy Power Demand dataset. Preprocess for train/test. Use Shapelet Transform (Accuray: 95%) and SAX-VSM (Accuray: 93%) to classify the 2 different time series.<p>

3. [Use Statistical methods & Machine Learning based approach for time series forecasting on short horizon: ](https://github.com/MNLepage08/YCNG-233/blob/main/Time%20Series%20-%20Course%204.ipynb)Collect Hydro-Quebec Dataset. Preprocess for duplicate(time change) & 0 values. Exploration result for find parameter of ARIMA Family (stationarity, trend, seasonality, ACF, PACF, History). Build SARIMA (sMAPE: 6.54), Holt-Winters (sMAPE: 6.13), Linear Regression (sMAPE: 6.69), Decision Tree (sMAPE: 8.19) and SVM (sMAPE: 6.84). Visualize the forecast result.
   
4. [Use Deep Feedforward Neural Network and LSTM model for time series forecasting on short horizon:]() Collect electricity consumption in New York. Preprocess to corect date format. Optimal window size selection with the pattern into the data and some experimentation. Build the architecture of these models by iteration to find the hyperparameter optimization. DNN: MAE = 1.0032, LSTM: MAE = 1.0070.

  
## :mortar_board: Courses

| # | Sessions |
| --- | --- |
| 1 | Introduction to Time Series problems |
| 2 | Time series classification |  
| 3 | Time series forecasting |
| 4 | Statistical methods for time series forecasting |
| 5 | Deep Learning for time series |
| 6 | Ensemble learning and its application on time series problems |
| 7 | Transfer learning and its application on time series problems |
| 8 | Graph neural networks and their applications on time series problems |
| 9 | Sequence models and their applications on time series problems |
| 10 | Project presentation |

  
## :pencil2: Notes
  
<details close>
<summary>1. Introduction to Time Series problems<p></summary>
  
* Time series: a suite of data point ordered over time. The minimal information in a time series is the datetime (or the timestamp) of the datapoint and the value itself. The value can be discrete or continuous.<p>
  
* UTC is a continuous value, but it might be hard to extract information such as the season, the night/day... For data science purpose, the preferred representation is a datetime + explicit time zone. [Datetime Library](https://docs.python.org/3/library/datetime.html) & [Pytz Library](https://pypi.org/project/pytz/)<p>
  
* [Time zones and offset: ](https://youtu.be/-5wpm-gesOY)The Problem with Time & Timezones<p>
  
* [Resample:](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html) You cannot assume data are equally distributer overtime. You must check it. Resampling will help to Handle duplicates and Highlight missing data.<p>
  
* Filling missing data: Many ways. The 2 mostly used are [Interpolation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html) and [Forward Fill](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html)<p>
  
* Just as correlation measures the extent of a linear relationship between two variables, autocorrelation measures the linear relationship between lagged values of a time series. The [autocorrelation function](https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.acf.html) describes the relationship between a time series and its lagged counterpart. The [partial autocorrelation](https://www.statsmodels.org/devel/generated/statsmodels.tsa.stattools.pacf.html) describes a direct relationship, that is, it removes the effects of the intermediate lagged values.<p>
  
* [Time Series Decomposition: ](https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/) Involves thinking of a series as a combination of level, trend, seasonality, and noise components.<p>
  
* [Stationary vs. Non-Stationary: ](https://machinelearningmastery.com/time-series-data-stationary-python/) A stationary series is one where the values of the series is not a function of time. Mean(ts) = Mean(slide), Variance(ts) = Variance(slide), Autocorrelation(ts) = Autocorrelation(slice). Test: Augmented Dickey–Fuller test.
  
</details>

<details close>
<summary>2. Time series classification<p></summary>
  
* Supervised Learning: $Dataset = {X,Y} (X = [x_1(t), ... ,x_N(t)], Y = [y_1, ... , y_N] | y_i$ is defined in a list of K classes containing N time series and N labels. The task is to find for a new time series x(t) the corresponding class. Classification problem.<p>
  
* Unsupervised or semi-supervised: $Dataset = {X}, X = [x_1(t), ... ,x_N(t)]$. The task is to detect if a new time series x(t) have a similar behavior than time series in Dataset. Anomaly detection.<p>
  
* Ontology: Feature based (x(t) --> Feature extraction --> Classifier --> Class), Distance based (x(t) --> Distance --> Class), Deep Learning (x(t), Classifier --> Class)<p>
  
* Feature based methods: The main idea is to extract relevant information from the time series and provide it to a classification algorithm. Simple feature ex: mean, variance, RMS. Energy/power features: Shannon entropy, coefficient from DFT (Discrete Fourier Transformation). Correlation features: number/position of the peaks in the autocorrelation... Limits: Features must be defined, High dimensionality, Non stationarity, Time structure is not considered.  [TsFresh Library](https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html)<p>
  
* Distance based methods: Shapelet, SAX, BOSS, BOSSVS, HIVE-COTE,  DTW. Pattern based and tolerate some flexibility on signal amplitude. [Pyts Library](https://pyts.readthedocs.io/en/stable/), [DTW Library](https://pypi.org/project/dtw-python/)
  
</details>

<details close>
<summary>3. Time series forecasting<p></summary>
  
* Concepts (Now, History, Step Size, Horizon): Now: the time where the prediction takes place. The “now” can be arbitrarily set. Each “now” will produce a new row. Step size: the time between 2 consecutives “now”s. Should be linked to the business problem. Do you need to do a prediction for each minute? Day? Month? History: For a given “now” how long in the past will you look at. Horizon: Number of steps in the future we would like to predict.<p>
  
* Preprocessing: Parse dates, Resample (sum / mean), Create X lags, Create Y outputs.<p>
  
* Evaluation metrucs: Mean absolute error (MAE), Root mean sqared error (RMSE), Mean absolute percentage error (MAPE), Symmetric mean absolute percentage error (sMAPE).<p>
  
* [Evaluation Strategy: ](https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/) k-fold Cross Validation Does Not Work For Time Series Data and Techniques That You Can Use Instead. Train-Test split that respect temporal order of observations. Multiple Train-Test splits that respect temporal order of observations. Walk-Forward Validation where a model may be updated each time step new data is received.<p>
  
* Simplest model (Benchmark): Prediction = mean of the time series, Prediction = the last seen value.<p>
  
* ARIMA Family model: No trend & seasonality -- AR(p), MA(q), ARMA(p, q). Trend with no seasonality -- ARIMA(p, d, q). Trend & sesasonality: SARIMA(p, d, q)(P, D, Q, s).<p>
  
* Exponential Smoothing: Suitable method when no clear trend or seasonality can be observed.
  
</details>
 
<details close>
<summary>4. Statistical methods for time series forecasting<p></summary>
  
* ARIMA RECAP: Look at the plot, Is there a trend?, Is there seasonality?, Compute ACP and PACP. Even if SARIMA is suitable for any situation, it is much harder to parametrize.<p>
  
* [Exponential Smoothing (ES): ](https://machinelearningmastery.com/exponential-smoothing-for-time-series-forecasting-in-python/) Single Exponential Smootthing (SES), Double Exponential Smoothing (Holt), Triple Exponential Smoothing (Holt-Winters).<p>
  
* Machine Learning models: Machine learning models extends the idea of an AR(p) model. P parameter is the size of the history. Build the dataset by sliding the “now”. Dimension of X = p, Dimension of output = h. Train any regression model. For a new “now”, provide the X values as the p-lags. Suited for multivariate time series...<p>
  
* [Introducing a New Hybrid ES-RNN Model ](https://www.uber.com/blog/m4-forecasting-competition/)<p>
  
* Recap: What is the granularity of the problem? Does the time series miss some data? Where the time series has been captured? What should be the size of the history? Does the time series have a trend? Does the time series have seasonality? Is the time series long or short? Is the time series multivariate? How many time series do I have to forecast?

</details>

<details close>
<summary>5. Deep Learning for time series<p></summary>

* M3, M4, M5 Competitions: Blind competition to Benchmark best time series forecasting methods. Each competition brought more attention / led to different conclusions.<p>
  
* [Statistical, machine learning and deep learning forecasting methods: Comparisons and ways forward](https://www.tandfonline.com/doi/full/10.1080/01605682.2022.2118629)<p>
  
* [Statistical and Machine Learning forecasting methods: Concerns and ways forward](https://www.researchgate.net/publication/323847484_Statistical_and_Machine_Learning_forecasting_methods_Concerns_and_ways_forward)<p>
  
* [The M3-Competition: results, conclusions and implications](https://www.sciencedirect.com/science/article/abs/pii/S0169207000000571?via%3Dihub). 3003 time series & 24 methods.<img width="887" alt="Capture d’écran, le 2023-06-03 à 18 39 15" src="https://github.com/MNLepage08/MNLepage08/assets/113123425/727490cc-b843-4363-97b5-94d8b7b054ee"><p>
  
* **M3 Conclusions:** Ensemble > Single methods. Short horizon (Statistical) vs. Long horizon (DL methods). Seasonality? (High (statistical) vs. Low (DL)). Statistical are very good. ML? No reason to spend time on it (didn't try lot of them...)<p>
  
* [GluonTS - Probabilistic Time Series Modeling in Python Librairy](https://ts.gluon.ai/stable/)<p>
  
* [The M4 Competition: ](https://www.sciencedirect.com/science/article/pii/S0169207019301128#fig1)100,000 time series and 61 forecasting methods. Most of the dataset has time series below 250 data points.<p>
  
* **M4 Conclusions:** Combination (ensemble) outperforms single methods (Statistical and Hybrid). Hybrid methods outperforms other methods. Pure ML doesn't work.<p>
  
* [M4 N-BEATS: NEURAL BASIS EXPANSION ANALYSIS FOR INTERPRETABLE TIME SERIES FORECASTING: ](https://www.researchgate.net/publication/333418084_N-BEATS_Neural_basis_expansion_analysis_for_interpretable_time_series_forecasting) Pure DL, 1 model per frequency. Ensemble of models.<p>
  
* M4 Ensemble weighted method (EWM): Helps with Few-Shot learning.<p>
  
* [N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting: ](https://arxiv.org/abs/2201.12886) Different datasets (“All large-scale datasets used in our empirical studies are publicly available and have been used in neural forecasting literature, particularly in the context of long-horizon”)<p>
  
* [M5 accuracy competition: Results, findings, and conclusions: ](https://www.sciencedirect.com/science/article/pii/S0169207021001874)42 000 hierarchical times series (Walmart) on Kaggle. Predict sales at different levels. For the first time, it focused on series that display intermittency, i.e., sporadic demand including zeros. Predict daily unit sales. Horizon = 28 days.<p>
  
* **M5 Conclusions:** LightGBM is superior. Ensemble methods are better. The external adjustments utilized in some methods were beneficial for improving the accuracy of the baseline forecasting models. Exogenous/explanatory variables were important for improving the forecasting accuracy of time series methods. Hierarchical is a different problem => top down, bottom up, middle out... still good results at the lowest level...<p>
  
* **Conclusions:** M3 - Statistical Approach, M4 - Deep Learning, M5 - Boosting. Ensemble learning is the best bet. Longer horizon, statistical approach fail. Complexity vs efficiency: boosting. Long forecast ? Only DL for now. Keep an eye on transformers.

</details>


<details close>
<summary>6. Ensemble learning and its application on time series problems<p></summary>

* [Darts](https://unit8co.github.io/darts/) is a Python library for user-friendly forecasting and anomaly detection on time series.<p>
  
* **Ensemble Learning & Ensemble Methods Inference:** <img width="800" alt="Capture d’écran, le 2023-06-08 à 13 40 25" src="https://github.com/MNLepage08/YCNG-228/assets/113123425/83648869-d5c1-4ac8-aca1-d9d9959a32f9"><p>

* <img width="500" align="right" alt="Capture d’écran, le 2023-06-08 à 15 23 39" src="https://github.com/MNLepage08/YCNG-228/assets/113123425/47372dca-012d-4845-b4c0-3ba11a2d095b">**Bootstrap Aggregating (Bagging):** Creates multiple overlapping (or not) subsets from the original dataset. Train a weak learner on each subset (can be done in parallel). Aggregate the prediction using an aggregation function. Can be expected: Bagginf is good to reduce variance be aware of overfitting. Often used with tree-based models (random forest). Solve the problem of instability (tiny difference in the feature space leads to huge differences). Naive Bayes classifiers or KNN classifiers are stable.<br><br><br><br>
  
* <img width="407" align="left" alt="Capture d’écran, le 2023-06-08 à 16 31 07" src="https://github.com/MNLepage08/YCNG-228/assets/113123425/62b18746-60b1-462b-8a02-67fd912ce208">**Boosting:** Create a week classifier. Look at misclassified data points. Increase the weight of those misclassified data point. Repeat for create a week classifier... [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html). [Gradient boosting:](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) don't change weight but train on residual errors.<p>

Boosting does more to reduce bias than variance. For this reason, boosting tends to improve upon its base models most when they have high bias and low variance. Boosting’s bias reduction comes from the way it adjusts its distribution over the training set. However, this method of adjusting the training set distribution causes boosting to have difficulty when the training data are noisy. Subsample parameter == bagging and boosting (never used).
  
--> MA Model & lightGBM. <p>
  
* [Tune lighGBM:](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html#tune-parameters-for-the-leaf-wise-best-first-tree) Parameters that affect the structure and learning of the decision trees, the training speed, for better accuracy, to cambat overfitting.<p>
  
* **Structure (complexity of the problem):** max_depth: max depth for each tree [3-12]. num_leaves: number of decision leaves in a single tree 2^(max_depth)[8-4096]. min_data_in_leaf: needs a certain amount of data to evaluate the leaf. Can be tune according the dataset size (tricky). **Accuracy:** Learning_rate: [0.01-0.3] (can be lower), decrease --> slower and more accurate. n_estimators: number of estimator, increase --> better accuracy + overfitting. increase n_estimators and decrease learning rate. **Control overfitting:** lambda_L1 and lambda_L2: [0-100]. min_grain_to_split: [0-15]. Bagging_fraction, feature_fraction: [0-1].<p>
  
* <img width="500" align="right" alt="Capture d’écran, le 2023-06-11 à 12 48 06" src="https://github.com/MNLepage08/YCNG-228/assets/113123425/faf9079b-8ba8-405c-9846-7cdf0ef4422e">[Staking: ](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html#sklearn-ensemble-stackingclassifier)In practice, base learners are different algorithms. The new dataset is often combined with the original dataset to train the Meta Model. The Meta-Learner is often a logistic regression but in theory, any algorithm can be used. Pretrained models can be used. This opens the door to transfer learning (next session). Can be used to augment the dataset and add external variables.<p>
  
* **Ensemble methods - Approach 1:** Models are trained on different view of the same dataset (Bagging, Boosting: man, median, vote). **Approach 2:** Models are trained on the same dataset but using different algoriths (Stacking). **Approach 3:** Models are trained on different datasets, selection of the top k models, train a Meta model. Ensemble weighted models.<p>
  
* **Continuous(/al) Learning**
 
</details>
  
## :books: Bibliography
| <img width="261" alt="Forecasting" src="https://github.com/MNLepage08/MNLepage08/assets/113123425/dd018b33-133b-496d-b1b7-1e89fee658c9">  | <img width="285" alt="Practical TS" src="https://github.com/MNLepage08/MNLepage08/assets/113123425/81b8c679-84c0-4179-8085-751da4c573e2"> | 
| :-------------: | :-------------: | 
| [Forecasting: Principles and Practice](https://otexts.com/fpp3/) | Practical Time Series Analysis| 
