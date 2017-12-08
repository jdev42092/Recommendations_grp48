---
title: Regularization Regression
notebook: Recommendations_Regularization.ipynb
nav_include: 3
---

## Contents
{:.no_toc}
*  
{: toc}





    /anaconda/lib/python3.6/site-packages/statsmodels/compat/pandas.py:56: FutureWarning: The pandas.core.datetools module is deprecated and will be removed in a future version. Please use the pandas.tseries module instead.
      from pandas.core import datetools



Now that we have our baseline results, we would like to run the actual fit of the model leaving parameters $\theta$ and $\gamma$ to be fit. We can do so by reconstructing the baseline model to the following:

$$ Y_{um} = \mu + \bar{\theta} \cdot I_u + \bar{\gamma} \cdot I_m$$

Where $\bar{\theta}$ is a vector of coefficients for users who have made ratings and $\bar{\gamma}$ is a coefficients for restaurants for which ratings have been made. We multiply these by indicator variables $I_u$ and $I_m$, respectively, for the u-th user and m-th restaurant to go in the feature matrix.  

The way we implement this is by constructing an $N$ by $U + M + 1$ matrix, where the $N$ is the number of reviews, $U$ is the total number of reviewers, and $M$ is the total number of restaurants (we include an additional column for the intercept).

We will run this matrix through a multiple linear regression to compare results with baseline method, but we will also run this matrix through Ridge and Lasso regularization, using both $R^2$, but also $RMSE$, as the data contains a lot of noise that will likely be overrepresented by using $R^2$. This should help compared to the linear regression run using this matrix process, as the number of features included in this regression has expanded greatly.

## Run regularization on full universe of reviews



```python
## Load in test and train data for all markets
train_df = pd.read_csv("Data/states/train/OH/train_150.csv")
test_df = pd.read_csv("Data/states/test/OH/test_150.csv")
print(train_df.shape)
print(test_df.shape)
```


    (3925, 13)
    (942, 13)




```python
train_small = train_df[['user_id','business_id','review_score']]
len(train_df.user_id.unique())
test_small = test_df[['user_id','business_id','review_score']]
```




```python
## Create user and business dummies in test and training set
train_dummies = pd.get_dummies(train_small, columns=['user_id','business_id'], drop_first=False)
test_dummies = pd.get_dummies(test_small, columns=['user_id','business_id'], drop_first=False)
```




```python
train_dummies.head()
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
      <th>review_score</th>
      <th>user_id_3Uv0dGI2IXJb2OUj8R2GJA</th>
      <th>user_id_5QFws6LKMCZCgKHl8WR1jQ</th>
      <th>user_id_CcOgdfEJxgrxTAwag5k18Q</th>
      <th>user_id_H_-K6erSJYtzg3ZEvOg3EQ</th>
      <th>user_id_NfU0zDaTMEQ4-X9dbQWd9A</th>
      <th>user_id_PrwnAL82LL4Ewt_wJpHWCA</th>
      <th>user_id_QaN-nccbLZPWzownQYgTVQ</th>
      <th>user_id_RlpkcJqctkKXl-LO1IAtig</th>
      <th>user_id_RylA6VZUTRuMGBu4nHKbCw</th>
      <th>...</th>
      <th>business_id_zW2Nzu38bB5nlOhhim-O5A</th>
      <th>business_id_zYbEKtLeosxhTzF4zSRIyA</th>
      <th>business_id_zc0sUY7iWuJB93AHWKy_xw</th>
      <th>business_id_zhBkNLn2KPnh5-NIueXVHA</th>
      <th>business_id_zl3Y1_DprpVzY3Izad4M-Q</th>
      <th>business_id_zlZQM-cJPVW7FHJsYTvyYg</th>
      <th>business_id_zluk4cL7Ch-uRlRply42ZQ</th>
      <th>business_id_zm3w7U26kDxREFDSLJRBgQ</th>
      <th>business_id_zo9fKM_Sty6qGztXKoMPmQ</th>
      <th>business_id_zzSYBWuv_fXGtSgsO-6_1g</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1739 columns</p>
</div>





```python
## Create matrices with only common users and businesses in them
train_cols = list(train_dummies.columns)
test_cols = list(test_dummies.columns)
all_cols = [x for x in train_cols if x in test_cols]
all_cols

train = train_dummies[all_cols]
test =test_dummies[all_cols]
print(train.shape)
print(test.shape)
```


    (3925, 530)
    (942, 530)




```python

```





    507





```python
## Create matrices with all users and all businesses in them, fill NaNs with 0s
train_cols = pd.DataFrame(columns = train_dummies.columns)
test_cols = pd.DataFrame(columns = test_dummies.columns)
all_cols = train_cols.append(test_cols)
all_cols

train = all_cols.append(train_dummies)
train = train.fillna(0.)
test = all_cols.append(test_dummies)
test = test.fillna(0.)
print(train.shape)
print(test.shape)
```


    (3925, 1938)
    (942, 1938)




```python
## Create train and test matrices for linear, Ridge, and Lasso regressions
X_train_all = train.drop('review_score', axis=1)
y_train_all = train['review_score']

X_test_all = test.drop('review_score', axis=1)
y_test_all = test['review_score']
```




```python
## Run matrices through linear regression
baseline_all = LinearRegression(fit_intercept=True)
baseline_all.fit(X_train_all, y_train_all)

#print('Baseline Intercept:', baseline_all.intercept_)
#print('Baseline Coefficients:', baseline_all.coef_)
print('Baseline Train Score:', baseline_all.score(X_train_all, y_train_all))
print('Baseline Test Score:', baseline_all.score(X_test_all, y_test_all))
print(sqrt(mean_squared_error(y_train_all, baseline_all.predict(X_train_all))))
print(sqrt(mean_squared_error(y_test_all, baseline_all.predict(X_test_all))))
```


    Baseline Train Score: 0.245120246601
    Baseline Test Score: -0.0631406527031
    0.8339818809755475
    1.0197984062853958


We see here that, because of all of the added factors for all users and all restaurants, this model is significantly overfitting to the training set. Our next set is regularization to correct for this overfitting>



```python
lambdas = [.001,.005,1,5,10,50,100,500,1000]

clf = RidgeCV(cv = 5, alphas=lambdas, fit_intercept=True)
clf.fit(X_train_all, y_train_all)
si= np.argsort(np.abs(clf.coef_))

print("----")
print('Ridge Train Score', clf.score(X_train_all, y_train_all))
print('Ridge Test Score', clf.score(X_test_all, y_test_all))
print(sqrt(mean_squared_error(y_train_all, clf.predict(X_train_all))))
print(sqrt(mean_squared_error(y_test_all, clf.predict(X_test_all))))

clfl = LassoCV(cv = 5, alphas=lambdas, fit_intercept=True)
clfl.fit(X_train_all, y_train_all)

print("----")
print('Lasso Train Score', clfl.score(X_train_all, y_train_all))
print('Lasso Test Score', clfl.score(X_test_all, y_test_all))
print(sqrt(mean_squared_error(y_train_all, clfl.predict(X_train_all))))
print(sqrt(mean_squared_error(y_test_all, clfl.predict(X_test_all))))
```


    ----
    Ridge Train Score 0.174564329138
    Ridge Test Score 0.0974114203465
    0.8720860570539439
    0.9396451650177554
    ----
    Lasso Train Score 0.094308804733
    Lasso Test Score 0.0523688975209
    0.9134984178347326
    0.9628056264554311


As we see, Ridge does much better than Lasso. This is because we do not want to zero out features, as is done in Lasso, we simply want to penalize the magnitudes of each coefficient. This method still turns out not to do as well as the baseline model.
