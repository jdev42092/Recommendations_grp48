---
title: Baseline Model
notebook: Recommendations_Baseline.ipynb
nav_include: 3
---

## Contents
{:.no_toc}
*  
{: toc}






At its most basic level, a recommendation system takes information previously collected on both users and items and is able to pair them together to predict how much a user would like new items. Often times, such a baseline is surprisingly effective, and improvements on the baseline are hard-won.

Here, we define our baseline using a simple multiple linear regression of average rating for each user and average rating for a given restaurant, $m$, to predict what each user would rate $m$. The model is as follows:

$$\hat{Y_{um}} = \hat{\mu} + \hat{\theta}_{u} + \hat{\gamma}_{m}$$

Where $$\hat{\theta}_{u}$$ is the average rating for user $u$, $$\hat{\gamma}_{m}$$ is the average rating for restaurant $m$, and
$$\hat{\mu}$$ is the intercept.


In our analysis, we will run this baseline model on ratings from Ohio, as it provides an inbetween representation of larger markets and smaller markets, and only with users that have at least 150 reviews in the review data sest provided by Yelp, as discussed on the EDA page.

## Run baseline model on full universe of reviews



```python
## Load in test and train data for all markets
#train_df = pd.read_csv("Data/train.csv")
#test_df = pd.read_csv("Data/test.csv")
train_df = pd.read_csv("Data/states/train/OH/train_150.csv")
test_df = pd.read_csv("Data/states/test/OH/test_150.csv")
print(train_df.shape)
print(test_df.shape)
```


    (3925, 13)
    (942, 13)




```python
train_df.head()
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review_date</th>
      <th>business_longitude</th>
      <th>business_id</th>
      <th>business_categories</th>
      <th>business_name</th>
      <th>business_state</th>
      <th>review_score</th>
      <th>user_id</th>
      <th>user_average_rating</th>
      <th>business_review_count</th>
      <th>business_average_rating</th>
      <th>business_latitude</th>
      <th>user_review_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2012-08-09</td>
      <td>-81.688974</td>
      <td>HNs2Nf-trqFTDtho4vhfmA</td>
      <td>['Bars', 'Lounges', 'Restaurants', 'American (...</td>
      <td>The South Side</td>
      <td>OH</td>
      <td>3.0</td>
      <td>3Uv0dGI2IXJb2OUj8R2GJA</td>
      <td>3.85</td>
      <td>275</td>
      <td>3.5</td>
      <td>41.482026</td>
      <td>482</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-09-27</td>
      <td>-81.579720</td>
      <td>SP7H3zPArNvbHKQW0c_gpA</td>
      <td>['Restaurants', 'Thai', 'Asian Fusion']</td>
      <td>High Thai'd</td>
      <td>OH</td>
      <td>2.0</td>
      <td>3Uv0dGI2IXJb2OUj8R2GJA</td>
      <td>3.85</td>
      <td>100</td>
      <td>4.0</td>
      <td>41.510991</td>
      <td>482</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-06-20</td>
      <td>-81.729861</td>
      <td>YgHp9MdZ1vVdYyMEro4TtQ</td>
      <td>['Bars', 'Barbeque', 'Pizza', 'American (New)'...</td>
      <td>XYZ the Tavern</td>
      <td>OH</td>
      <td>4.0</td>
      <td>3Uv0dGI2IXJb2OUj8R2GJA</td>
      <td>3.85</td>
      <td>181</td>
      <td>4.0</td>
      <td>41.484139</td>
      <td>482</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014-04-16</td>
      <td>-81.730410</td>
      <td>wmstf9dw0-kN3YThIxx8eQ</td>
      <td>['Irish', 'Bars', 'Pubs', 'Nightlife', 'Restau...</td>
      <td>Stone Mad Pub</td>
      <td>OH</td>
      <td>4.0</td>
      <td>3Uv0dGI2IXJb2OUj8R2GJA</td>
      <td>3.85</td>
      <td>126</td>
      <td>3.5</td>
      <td>41.486707</td>
      <td>482</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2010-08-02</td>
      <td>-81.690048</td>
      <td>Xny0n0s98TpP82sQxfgIMQ</td>
      <td>['Polish', 'Nightlife', 'Restaurants', 'Americ...</td>
      <td>Sokolowski's University Inn</td>
      <td>OH</td>
      <td>3.0</td>
      <td>3Uv0dGI2IXJb2OUj8R2GJA</td>
      <td>3.85</td>
      <td>368</td>
      <td>4.5</td>
      <td>41.484752</td>
      <td>482</td>
    </tr>
  </tbody>
</table>
</div>





```python
print(train_df.columns)
train_df.describe()
```


    Index(['review_date', 'business_longitude', 'business_id',
           'business_categories', 'business_name', 'business_state',
           'review_score', 'user_id', 'user_average_rating',
           'business_review_count', 'business_average_rating', 'business_latitude',
           'user_review_count'],
          dtype='object')





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>business_longitude</th>
      <th>review_score</th>
      <th>user_average_rating</th>
      <th>business_review_count</th>
      <th>business_average_rating</th>
      <th>business_latitude</th>
      <th>user_review_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3925.000000</td>
      <td>3925.000000</td>
      <td>3925.000000</td>
      <td>3925.000000</td>
      <td>3925.000000</td>
      <td>3925.000000</td>
      <td>3925.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-81.671409</td>
      <td>3.696051</td>
      <td>3.826395</td>
      <td>116.180637</td>
      <td>3.691338</td>
      <td>41.470231</td>
      <td>731.208662</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.142339</td>
      <td>0.960004</td>
      <td>0.192418</td>
      <td>133.248942</td>
      <td>0.598440</td>
      <td>0.076287</td>
      <td>439.526213</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-82.226472</td>
      <td>1.000000</td>
      <td>3.320000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>41.108641</td>
      <td>309.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-81.761182</td>
      <td>3.000000</td>
      <td>3.700000</td>
      <td>33.000000</td>
      <td>3.500000</td>
      <td>41.459052</td>
      <td>464.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-81.690421</td>
      <td>4.000000</td>
      <td>3.780000</td>
      <td>74.000000</td>
      <td>4.000000</td>
      <td>41.484801</td>
      <td>609.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>-81.580318</td>
      <td>4.000000</td>
      <td>3.970000</td>
      <td>148.000000</td>
      <td>4.000000</td>
      <td>41.500613</td>
      <td>868.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>-81.072826</td>
      <td>5.000000</td>
      <td>4.170000</td>
      <td>896.000000</td>
      <td>5.000000</td>
      <td>41.764307</td>
      <td>1952.000000</td>
    </tr>
  </tbody>
</table>
</div>





```python
X_train_all = train_df[['user_average_rating', 'business_average_rating']]
y_train_all = train_df['review_score']
X_test_all = test_df[['user_average_rating', 'business_average_rating']]
y_test_all = test_df['review_score']
```




```python
baseline_all = LinearRegression(fit_intercept=True)
baseline_all.fit(X_train_all, y_train_all)

print('Baseline Intercept:', baseline_all.intercept_)
print('Baseline Coefficients:', baseline_all.coef_)
print('Baseline Train Score:', baseline_all.score(X_train_all, y_train_all))
print('Baseline Test Score:', baseline_all.score(X_test_all, y_test_all))
print(sqrt(mean_squared_error(y_train_all, baseline_all.predict(X_train_all))))
print(sqrt(mean_squared_error(y_test_all, baseline_all.predict(X_test_all))))
```


    Baseline Intercept: -1.71779400589
    Baseline Coefficients: [ 0.81630482  0.62046353]
    Baseline Train Score: 0.196211027717
    Baseline Test Score: 0.189254189855
    0.8605750453011812
    0.8905561821699436




```python

```
