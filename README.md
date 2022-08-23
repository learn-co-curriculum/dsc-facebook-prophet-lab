# Facebook Prophet - Lab

## Introduction

In the last lab, you learned how to visualize and manipulate time series data, and how to use ARIMA modeling to produce forecasts for time series data. You also learned how to conclude a correct parametrization of ARIMA models. This can be a complicated process, and while statistical programming languages such as R provide automated ways to solve this issue, those have yet to be officially ported over to Python. 

Fortunately, the Data Science team at Facebook recently published a new library called `prophet`, which enables data analysts and developers alike to perform forecasting at scale in Python. We encourage you to read [this article](https://research.facebook.com/blog/2017/2/prophet-forecasting-at-scale/) by Facebook explaining how `prophet` simplifies the forecasting process and provides an improved predictive ability. 

## Objectives

- Model a time series using the Facebook's Prophet 
- Describe the difference between ARIMA and Additive Synthesis for time series forecasting 
- Use the methods in the `prophet` library to plot predicted values 

## Facebook's Prophet

Facebook's `prophet` uses an elegant yet simple method for analyzing and predicting periodic data known as the **additive modeling**. The idea is straightforward: represent a time series as a combination of patterns at different scales such as daily, weekly, seasonally, and yearly, along with an overall trend. Your energy use might rise in the summer and decrease in the winter, but have an overall decreasing trend as you increase the energy efficiency of your home. An additive model can show us both patterns/trends and make predictions based on these observations.

The following image shows an additive model decomposition of a time series into an overall trend, yearly trend, and weekly trend.

![additive model image from Facebook blog post](https://scontent-lga3-1.xx.fbcdn.net/v/t39.8562-6/240830073_526712221759967_1137977873639917627_n.png?_nc_cat=103&ccb=1-7&_nc_sid=6825c5&_nc_ohc=G52vWPIYHVYAX_JbvbH&_nc_ht=scontent-lga3-1.xx&oh=00_AT-RbQfa68-tBqQhZdpVSUbOiv-t1N6xq9jfqhSGzlRmmQ&oe=630A5A8B)

*“Prophet has been a key piece to improving Facebook’s ability to create a large number of trustworthy forecasts used for decision-making and even in product features.”*

Let's start by reading in our time series data. We will cover some data manipulation using `pandas`, accessing financial data using the `Quandl` library, and plotting with `matplotlib`. 


```python
# Import necessary libraries
import warnings
warnings.filterwarnings('ignore')

import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
%matplotlib inline
from matplotlib.pylab import rcParams
plt.style.use('fivethirtyeight')

from prophet import Prophet
```

    Importing plotly failed. Interactive plots will not work.



```python
# Import passengers.csv and set it as a time series
ts = pd.read_csv('passengers.csv')
ts['Month'] = pd.DatetimeIndex(ts['Month'])
```

The `prophet` library also imposes the strict condition that the input columns be named `ds` (the time column) and `y` (the metric column), so let's rename the columns in our `ts` DataFrame. 


```python
# Rename the columns [Month, AirPassengers] to [ds, y]
ts = ts.rename(columns={'Month': 'ds',
                        '#Passengers': 'y'})

ts.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ds</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1949-01-01</td>
      <td>112</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1949-02-01</td>
      <td>118</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1949-03-01</td>
      <td>132</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1949-04-01</td>
      <td>129</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1949-05-01</td>
      <td>121</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot the timeseries 
pd.plotting.register_matplotlib_converters()
ax = ts.set_index('ds').plot(figsize=(15, 8))
ax.set_ylabel('No. of Airline Passengers/Month')
ax.set_xlabel('Date')

plt.show()
```


    
![png](index_files/index_6_0.png)
    


## Time Series Forecasting with Prophet

We will now learn how to use the `prophet` library to predict future values of our time series. The Facebook team has abstracted away many of the inherent complexities of time series forecasting and made it more intuitive for analysts and developers alike to work with time series data.

To begin, we will create a new prophet object with `Prophet()` and provide a number of arguments. For example, we can specify the desired range of our uncertainty interval by setting the `interval_width` parameter.


```python
# Set the uncertainty interval to 95% (the Prophet default is 80%)
Model = Prophet(interval_width=0.95)
```

Now that our model has been initialized, we can call its `.fit()` method with our DataFrame `ts` as input. The model fitting should take no longer than a few seconds.


```python
# Fit the timeseries to Model
Model.fit(ts)
```

    13:22:13 - cmdstanpy - INFO - Chain [1] start processing
    13:22:13 - cmdstanpy - INFO - Chain [1] done processing





    <prophet.forecaster.Prophet at 0x40749ca080>



In order to obtain forecasts of our time series, we must provide the model with a new dataframe containing a `ds` column that holds the dates for which we want predictions. Conveniently, we do not have to concern ourselves with manually creating this dataframe because prophet provides the `.make_future_dataframe()` helper method. We will call this function to generate 36 datestamps in the future. The documentation for this method is available [here](https://www.rdocumentation.org/packages/prophet/topics/make_future_dataframe).

It is also important to consider the frequency of our time series. Because we are working with monthly data, we clearly specified the desired frequency of the timestamps (in this case, MS is the start of the month). Therefore, the `.make_future_dataframe()` will generate 36 monthly timestamps for us. In other words, we are looking to predict future values of our time series 3 years into the future.


```python
# Use make_future_dataframe() with a monthly frequency and periods = 36 for 3 years
future_dates = Model.make_future_dataframe(periods=36, freq='MS')
future_dates.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ds</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>175</th>
      <td>1963-08-01</td>
    </tr>
    <tr>
      <th>176</th>
      <td>1963-09-01</td>
    </tr>
    <tr>
      <th>177</th>
      <td>1963-10-01</td>
    </tr>
    <tr>
      <th>178</th>
      <td>1963-11-01</td>
    </tr>
    <tr>
      <th>179</th>
      <td>1963-12-01</td>
    </tr>
  </tbody>
</table>
</div>



This future dates dataframe can now be used as input to the `.predict()` method of the fitted model.


```python
# Predict the values for future dates and take the head of forecast
forecast = Model.predict(future_dates)
forecast.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ds</th>
      <th>trend</th>
      <th>yhat_lower</th>
      <th>yhat_upper</th>
      <th>trend_lower</th>
      <th>trend_upper</th>
      <th>additive_terms</th>
      <th>additive_terms_lower</th>
      <th>additive_terms_upper</th>
      <th>yearly</th>
      <th>yearly_lower</th>
      <th>yearly_upper</th>
      <th>multiplicative_terms</th>
      <th>multiplicative_terms_lower</th>
      <th>multiplicative_terms_upper</th>
      <th>yhat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1949-01-01</td>
      <td>106.199036</td>
      <td>41.687583</td>
      <td>130.259295</td>
      <td>106.199036</td>
      <td>106.199036</td>
      <td>-21.964653</td>
      <td>-21.964653</td>
      <td>-21.964653</td>
      <td>-21.964653</td>
      <td>-21.964653</td>
      <td>-21.964653</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>84.234383</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1949-02-01</td>
      <td>108.385288</td>
      <td>32.799167</td>
      <td>121.682585</td>
      <td>108.385288</td>
      <td>108.385288</td>
      <td>-30.742163</td>
      <td>-30.742163</td>
      <td>-30.742163</td>
      <td>-30.742163</td>
      <td>-30.742163</td>
      <td>-30.742163</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>77.643125</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1949-03-01</td>
      <td>110.359967</td>
      <td>68.431103</td>
      <td>154.167749</td>
      <td>110.359967</td>
      <td>110.359967</td>
      <td>-0.494234</td>
      <td>-0.494234</td>
      <td>-0.494234</td>
      <td>-0.494234</td>
      <td>-0.494234</td>
      <td>-0.494234</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>109.865733</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1949-04-01</td>
      <td>112.546219</td>
      <td>63.118888</td>
      <td>154.836731</td>
      <td>112.546219</td>
      <td>112.546219</td>
      <td>-5.201420</td>
      <td>-5.201420</td>
      <td>-5.201420</td>
      <td>-5.201420</td>
      <td>-5.201420</td>
      <td>-5.201420</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>107.344799</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1949-05-01</td>
      <td>114.661946</td>
      <td>67.628286</td>
      <td>155.423695</td>
      <td>114.661946</td>
      <td>114.661946</td>
      <td>-3.802447</td>
      <td>-3.802447</td>
      <td>-3.802447</td>
      <td>-3.802447</td>
      <td>-3.802447</td>
      <td>-3.802447</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>110.859499</td>
    </tr>
  </tbody>
</table>
</div>



We can see that Prophet returns a large table with many interesting columns, but we subset our output to the columns most relevant to forecasting, which are:

* `ds`: the datestamp of the forecasted value
* `yhat`: the forecasted value of our metric (in Statistics, yhat is a notation traditionally used to represent the predicted values of a value y)
* `yhat_lower`: the lower bound of our forecasts
* `yhat_upper`: the upper bound of our forecasts


```python
# Subset above mentioned columns and view the tail 
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ds</th>
      <th>yhat</th>
      <th>yhat_lower</th>
      <th>yhat_upper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>175</th>
      <td>1963-08-01</td>
      <td>649.244391</td>
      <td>605.894058</td>
      <td>696.349120</td>
    </tr>
    <tr>
      <th>176</th>
      <td>1963-09-01</td>
      <td>601.696467</td>
      <td>557.427798</td>
      <td>647.235297</td>
    </tr>
    <tr>
      <th>177</th>
      <td>1963-10-01</td>
      <td>565.653369</td>
      <td>518.798873</td>
      <td>611.856044</td>
    </tr>
    <tr>
      <th>178</th>
      <td>1963-11-01</td>
      <td>533.659919</td>
      <td>490.974144</td>
      <td>582.956115</td>
    </tr>
    <tr>
      <th>179</th>
      <td>1963-12-01</td>
      <td>563.257635</td>
      <td>520.245793</td>
      <td>607.852477</td>
    </tr>
  </tbody>
</table>
</div>



A variation in values from the output presented above is to be expected as Prophet relies on Markov chain Monte Carlo (MCMC) methods to generate its forecasts. MCMC is a stochastic process, so values will be slightly different each time.

Prophet also provides a convenient method to quickly plot the results of our forecasts.


```python
# Use Prophet's plot method to plot the predictions
Model.plot(forecast, uncertainty=True)
plt.show()
```


    
![png](index_files/index_18_0.png)
    


Prophet plots the observed values of the time series (the black dots), the forecasted values (blue line) and the uncertainty intervals of our forecasts (the blue shaded regions).

One other particularly strong feature of Prophet is its ability to return the components of our forecasts. This can help reveal how daily, weekly, and yearly patterns of the time series contribute to the overall forecasted values. We can use the `.plot_components()` method to view the individual components.


```python
# Plot model components 
Model.plot_components(forecast)
plt.show()
```


    
![png](index_files/index_20_0.png)
    


Since we are working with monthly data, Prophet will plot the trend and the yearly seasonality but if you were working with daily data, you would also see a weekly seasonality plot included. 

From the trend and seasonality, we can see that the trend is playing a large part in the underlying time series and seasonality comes into play mostly toward the beginning and the end of the year. With this information, we've been able to quickly model and forecast some data to get a feel for what might be coming our way in the future from this particular dataset. 

## Summary 
In this lab, you learned how to use the `prophet` library to perform time series forecasting in Python. We have been using out-of-the box parameters, but Prophet enables us to specify many more arguments. In particular, Prophet provides the functionality to bring your own knowledge about time series to the table.
