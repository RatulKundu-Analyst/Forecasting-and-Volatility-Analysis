# Abstract

Predicting stock prices is a complex and challenging problem that requires the use of various methods and techniques to capture the dynamic and stochastic nature of the market. One of the most common methods for forecasting stock prices is time series analysis, which can model the temporal patterns and trends in the data. However, time series analysis alone may not be able to account for the high volatility and uncertainty of the stock market, which are affected by many external factors such as news, events, sentiments, and expectations. Therefore, it is useful to combine time series analysis with other methods that can model the variability and heteroskedasticity of the stock returns.
The aim of this project is to present a comprehensive framework for predicting stock prices using time series analysis methods for daily price index of STATE BANK of INDIA, such as ARIMA model, and volatility analysis methods, such as ARCH, GARCH and its variants. We first use the ARIMA model to capture the mean equation of the stock returns, which can account for the dynamics, trend and random noise in the data. Then we use the ARCH and GARCH model to model the conditional variance of the stock returns, which can reflect the volatility clustering and persistence that are common in financial data. Volatility clustering refers to the tendency of high or low volatility periods to follow each other, creating patterns of variation over time. Persistence refers to the slow decay of the impact of a shock to the volatility, implying a long memory in the variance process. ARCH and GARCH models can incorporate these features by allowing the conditional variance to depend on past values of the series or past values of the variance itself. 
We also explore different extensions of the GARCH model, such as ARMA-GARCH and EGARCH, to compare their performance and suitability for prediction. We assess our framework based on their AIC and RMSE, which are criteria that measure the fit and the prediction accuracy of a model after it has been trained. We also compare the models among themselves to select the best model for our stock price prediction. 
To achieve this, we first check for Stationarity and Autocorrelation of the data and only proceed with the modelling after satisfying these assumptions (with or without transformation). We then train the models to find the best fitted model using AIC. Using the best fitted model, we make predictions and test their accuracy using RMSE. We summarize the results of the analysis and the performance of the predictive models. We also provide insights into the potential future trends and risks associated with the stock price based on the predictive models’ outcomes.	

# Methodology

	DATA COLLECTION & PREPROCESSING: - 
At first, we collected the data consulting with our guide Dr. Chiranjib Neogi from an website Investing.com . This dataset contains an information on 5093 observations with variables such as Date, Price, Open, High, Low, Volume, and Change. We prepared the data for analysis by cleaning and pre-processing as necessary.

	EXPLORATORY DATA ANALYSIS (EDA): - 
EDA stands for Exploratory Data Analysis. It is a critical initial step in the data analysis process, where data is visually and statistically explored to understand its main characteristics, uncover patterns, identify outliers, and gain insights. EDA involves techniques like data visualization, summary statistics, and data transformation to prepare the data for further analysis and modelling. 

	MODEL FITTING & EVALUATION: -
Model fitting is a process of adjusting the parameters of a financial model to make it more accurate and generalizable. It is an important step in building and evaluating financial models, as it helps to find the optimal balance between complexity and simplicity. A well-fit model can capture the patterns and relationships in the data without over fitting or under fitting. 
There are different financial models which are used to analyse financial series, such as:
-  ARMA (Autoregressive Moving Average): An Autoregressive Moving Average Model is a statistical model that uses past values and past errors of a time series to predict its future behaviour.
-  ARIMA (Autoregressive Integrated Moving Average):  An Autoregressive Integrated Moving Average Model is a generalization of an ARMA model which combines the AR and MA components and also includes an integration component, which transforms the non-stationary property to stationary property. An ARIMA model can capture various patterns and trends in time series data, such as seasonality, cyclicity or shocks. 

We will evaluate these models based on their predictive accuracy, root mean square error (RMSE) on the test dataset, and their Akaike Information Criterion (AIC) on the train dataset. AIC and BIC are methods to select a model in statistics based on its fit to the data and its complexity. They are calculated from log-likelihood and a penalty term that increases with the number of parameters in the model. They are better than other information criteria because they have some theoretical and practical advantages.
	Some of the advantages of AIC and BIC are:

	•	They are easy to compute and interpret, as they only depend on the log-likelihood and the number of parameters of the model and can be applied to any model that has a likelihood function, regardless of the distributional assumptions or the estimation method.

	•	They have asymptotic properties that guarantee their consistency and efficiency under certain conditions. They balance the trade-off between model fit and model complexity, by penalizing models that have more parameters. This helps to avoid overfitting, which is when a model fits the data too well but fails to generalize to new data.
But in these paper as both of them produces approximately same results, so we will only use AIC .

AIC = -2ln(L)  + 2k

Where L is the value of the likelihood function evaluated at the parameter estimates, and k is the number of estimated parameters. 
And RMSE tells us about the average distance between the residuals. The lower the RMSE, the better the model.
 
 		RMSE=√((∑_i(P_i-O_i )^2 )/n) 		i=1(1)n		

,where:	

Pi is the predicted value for the ith observation in the dataset.

Oi is the observed value for the ith  observation in the dataset.

n is the sample size

 
 	VOLATILITY FORECASTING: -

Volatility forecasting is a crucial aspect of financial analysis, risk management, and investment decision-making. It involves predicting the future volatility of financial assets, such as stocks, currencies, or commodities, based on historical price movements. Accurate volatility forecasts aid investors and traders in assessing market risk, constructing portfolios, and implementing hedging strategies.

Various statistical and econometric models are employed for volatility forecasting, including:

	ARCH (Autoregressive Conditional Heteroskedasticity) Models: 
            An AutoRegressive Conditional Heteroskedasticity Model is a statistical Model that predicts the volatility of a time series based on its past values and errors. An ARCH Model can capture the volatility persistence and asymmetry in time series data.

	GARCH (Generalized Autoregressive Conditional Heteroskedasticity) Models:  
            A Generalisation of ARCH model is the Generalized Autoregressive Conditional Heteroskedasticity model, which includes both autoregressive and moving average term in the variance equation. GARCH model can capture volatility clustering and account for nonuniform variance in time series data. There are several forms in GARCH modelling such as GARCH (1, 1).

	ARMA-GARCH Models: 
            ARMA-GARCH is a combination of two statistical models, ARMA and GARCH, that can be used two analyse and forecast time series data. These models can capture the dynamics of the mean and the volatility of a time series, respectively.

	E-GARCH Models: 
             E-GARCH stands for Exponential Generalized Autoregressive Conditional Heteroskedasticity, which means the variance of the series depends on its past values and errors, as well as their signs and magnitudes. E-GARCH models are useful for capturing the asymmetric effects of positive and negative shocks on the volatility, as well as the persistence and clustering of volatility over time.

It is essential to evaluate and compare different forecasting methods to select the one that best fits the characteristics of the financial time series data. Volatility forecasting is subject to uncertainty, and models may require regular updating and recalibration as new data becomes available. Investors and analysts should exercise caution and use volatility forecasts as one of several tools in their decision-making processes.
