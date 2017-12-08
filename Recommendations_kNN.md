---
title: Pearsons R/kNN Distance-Based Model
notebook: Recommendations_kNN.ipynb
nav_include: 5
---

## Contents
{:.no_toc}
*  
{: toc}

**Using k-Nearest Neighbors to Identify User Ratings**

This particular model uses a concept called neighborhood collaborative filtering to identify a small number of recommended restaurants for a particular user based on the same user's previously-stated preferences for similar restaurants. As was previously mentioned, the sample we are using for this model includes only those reviewers who have reviewed at least 150 restaurants previously, and thus the stated preferences are already present in the sample used for this model.

The model included here is based on a solution to the same problem for CS109a in 2013. The documentation for this problem can be found here: http://nbviewer.jupyter.org/github/cs109/content/blob/master/HW4_solutions.ipynb






**Read in Data**

The training and test samples used here were created previously prior to beginning analysis. They are the same training and test sets as have been used in previous models throughout this project.



```python
train_data = pd.read_csv('Data/train/OH/train_150.csv')
test_data = pd.read_csv('Data/test/OH/test_150.csv')
```




```python
train_data.shape, test_data.shape
```





    ((3925, 13), (942, 13))





```python
train_data.head()
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



**Data Cleaning**

Because this model is based on a user's previous experiences with similar restaurants, we need a way to define which restaurants in this dataset are similar to one another. One such measure is a "common user support", which shows the number of users who have rated any particular pair of restaurants. We need such a measurement because common user support can be used later throughout this problem as a proxy for how similar a pair of restaurants may be to each other.  



```python
restaurants=train_data.business_id.unique()
supports=[]
for i,rest1 in enumerate(restaurants):
    for j,rest2 in enumerate(restaurants):
        if  i < j:
            rest1_reviewers = train_data[train_data.business_id==rest1].user_id.unique()
            rest2_reviewers = train_data[train_data.business_id==rest2].user_id.unique()
            common_reviewers = set(rest1_reviewers).intersection(rest2_reviewers)
            supports.append(len(common_reviewers))
print("Mean support is:",np.mean(supports))
plt.hist(supports)
```


    Mean support is: 0.259574439166





    (array([  1.42330300e+06,   3.55450000e+04,   8.73300000e+03,
              2.49000000e+03,   8.62000000e+02,   3.44000000e+02,
              1.31000000e+02,   4.20000000e+01,   1.70000000e+01,
              3.00000000e+00]),
     array([  0. ,   1.1,   2.2,   3.3,   4.4,   5.5,   6.6,   7.7,   8.8,
              9.9,  11. ]),
     <a list of 10 Patch objects>)




![png](Recommendations_kNN_files/Recommendations_kNN_7_2.png)


On average, we find that there are very few pairs of restaurants that share a reviewer. We will disregard this to continue on with the creation of this model.

**Create Database**

Now that we have defined similar restaurants, we use this information to create a database of information about each pair of restaurants. The following information is saved:

- Rho: correlation measure that determines the correlation coefficient between the users' average ratings for each restaurant pairing. As an edit to the original referenced solution, this function will return a rho (Pearson's similarity index) equal to 0 when the variance in a particular set of reviews is equal to 0 for a particular pair of restaurants. Created in the "pearson_sim" function.

- Restaurant reviews: A set of reviews given a particular restaurant and a shared set of reviewers. Created in the "get_restaurant_reviews" function.

- Sim: A similarity of any two sets of restaurants given a shared set of reviewers and the reviews of those users. Created in the "calculate_similarity" function.

All of this information is wrapped up into a database called "db", which holds information about each pair of businesses in the dataset for Ohio.



```python
from scipy.stats.stats import pearsonr
def pearson_sim(rest1_reviews, rest2_reviews, n_common):
    """
    Given a subframe of restaurant 1 reviews and a subframe of restaurant 2 reviews,
    where the reviewers are those who have reviewed both restaurants, return
    the pearson correlation coefficient between the user average subtracted ratings.
    The case for zero common reviewers is handled separately. Its
    ok to return a NaN if any of the individual variances are 0.
    """
    if n_common==0:
        rho=0.
    else:
        diff1=rest1_reviews['business_average_rating']-rest1_reviews['user_average_rating']
        diff2=rest2_reviews['business_average_rating']-rest2_reviews['user_average_rating']
        try:
            rho=pearsonr(diff1, diff2)[0]
        except:
            return 0
    return rho
```




```python
def get_restaurant_reviews(restaurant_id, df, set_of_users):
    """
    given a resturant id and a set of reviewers, return the sub-dataframe of their
    reviews.
    """
    mask = (df.user_id.isin(set_of_users)) & (df.business_id==restaurant_id)
    reviews = df[mask]
    reviews = reviews[reviews.user_id.duplicated()==False]
    return reviews
```




```python
def calculate_similarity(rest1, rest2, df, similarity_func):
    # find common reviewers
    rest1_reviewers = df[df.business_id==rest1].user_id.unique()
    rest2_reviewers = df[df.business_id==rest2].user_id.unique()
    common_reviewers = set(rest1_reviewers).intersection(rest2_reviewers)
    n_common=len(common_reviewers)
    #get reviews
    rest1_reviews = get_restaurant_reviews(rest1, df, common_reviewers)
    rest2_reviews = get_restaurant_reviews(rest2, df, common_reviewers)
    sim=similarity_func(rest1_reviews, rest2_reviews, n_common)
    if np.isnan(sim):
        return 0, n_common
    return sim, n_common
```




```python
class Database:


    def __init__(self, df):

        database={}
        self.df=df
        self.uniquebizids={v:k for (k,v) in enumerate(df.business_id.unique())}
        keys=self.uniquebizids.keys()
        l_keys=len(keys)
        self.database_sim=np.zeros([l_keys,l_keys])
        self.database_sup=np.zeros([l_keys, l_keys], dtype=np.int)

    def populate_by_calculating(self, similarity_func):

        items=self.uniquebizids.items()
        for b1, i1 in items:
            for b2, i2 in items:
                if i1 < i2:
                    sim, nsup=calculate_similarity(b1, b2, self.df, similarity_func)
                    self.database_sim[i1][i2]=sim
                    self.database_sim[i2][i1]=sim
                    self.database_sup[i1][i2]=nsup
                    self.database_sup[i2][i1]=nsup
                elif i1==i2:
                    nsup=self.df[self.df.business_id==b1].user_id.count()
                    self.database_sim[i1][i1]=1.
                    self.database_sup[i1][i1]=nsup


    def get(self, b1, b2):

        sim=self.database_sim[self.uniquebizids[b1]][self.uniquebizids[b2]]
        nsup=self.database_sup[self.uniquebizids[b1]][self.uniquebizids[b2]]
        return (sim, nsup)
```




```python
np.seterr(all='raise')
db=Database(train_data)
db.populate_by_calculating(pearson_sim)
```


**Cleaning the Database**

The function "shrunk_sim" regularizes the similarity index used in the above database to make it easier to work with, particularly since we are working in a state with such a low number of shared users fore each pair of restaurants. This lessens the effect of small common supports.



```python
def shrunk_sim(sim, n_common, reg=3.):
    "takes a similarity and shrinks it down by using the regularizer"
    ssim=(n_common*sim)/(n_common+reg)
    return ssim
```


**Creating a kNN prediction of user preferences**

Based on the information in this similarity index, we can find use this to find similar neighbors, making it easier to use a kNN regression to find predictions.

This process is led by a number of functions that do the following:

- knearest: Creates a sorted list of similar restaurants given a particular restaurant.
- get_user_top_choices: Retrieves the top 5 restaurants for a user based on a user's star ratings.
- get_top_recos_for_user: Retrieves a sorted list of a user's recommendations, using the results of the "knearest" function above
- knearest_among_userrated: Returns a list of the best recommendations of restaurants given a user's previous rated.
- rating: Returns the ratings for those best recommendations of restaurants
- find_RMSE: Finds out how well you predicted, for both your test and train data! (Uses RMSE as a measure of predictability)



```python
from operator import itemgetter
def knearest(restaurant_id, set_of_restaurants, dbase, k=7, reg=3.):

    similars=[]
    for other_rest_id in set_of_restaurants:
        if other_rest_id!=restaurant_id:
            sim, nc=dbase.get(restaurant_id, other_rest_id)
            ssim=shrunk_sim(sim, nc, reg=reg)
            similars.append((other_rest_id, ssim, nc ))
    similars=sorted(similars, key=itemgetter(1), reverse=True)
    return similars[0:k]
```




```python
def get_user_top_choices(user_id, df, numchoices=5):
    udf=df[df.user_id==user_id][['business_id','review_score']].head(numchoices)
    return udf
```




```python
def get_top_recos_for_user(userid, df, dbase, n=5, k=7, reg=3.):
    bizs=get_user_top_choices(userid, df, numchoices=n)['business_id'].values
    rated_by_user=df[df.user_id==userid].business_id.values
    tops=[]
    for ele in bizs:
        t=knearest(ele, df.business_id.unique(), dbase, k=k, reg=reg)
        for e in t:
            if e[0] not in rated_by_user:
                tops.append(e)

    #there might be repeats. unique it
    ids=[e[0] for e in tops]
    uids={k:0 for k in list(set(ids))}

    topsu=[]
    for e in tops:
        if uids[e[0]] == 0:
            topsu.append(e)
            uids[e[0]] =1
    topsr=[]     
    for r, s,nc in topsu:
        avg_rate=df[df.business_id==r].review_score.mean()
        topsr.append((r,avg_rate))

    topsr=sorted(topsr, key=itemgetter(1), reverse=True)

    if n < len(topsr):

        return topsr[0:n]
    else:

        return topsr
```




```python
def knearest_amongst_userrated(restaurant_id, user_id, df, dbase, k=7, reg=3.):
    dfuser=df[df.user_id==user_id]
    bizsuserhasrated=dfuser.business_id.unique()
    return knearest(restaurant_id, bizsuserhasrated, dbase, k=k, reg=reg)
```




```python
def rating(df, dbase, restaurant_id, user_id, k=7, reg=3.):
    mu=df.review_score.mean()
    users_reviews=df[df.user_id==user_id]
    nsum=0.
    scoresum=0.
    nears=knearest_amongst_userrated(restaurant_id, user_id, df, dbase, k=k, reg=reg)
    restaurant_mean=df[df.business_id==restaurant_id].business_average_rating.values[0]
    user_mean=users_reviews.user_average_rating.values[0]
    scores=[]
    for r,s,nc in nears:
        scoresum=scoresum+s
        scores.append(s)
        r_reviews_row=users_reviews[users_reviews['business_id']==r]
        r_stars=r_reviews_row.review_score.values[0]
        r_avg=r_reviews_row.business_average_rating.values[0]
        rminusb=(r_stars - (r_avg + user_mean - mu))
        nsum=nsum+s*rminusb
    baseline=(user_mean +restaurant_mean - mu)
    #we might have nears, but there might be no commons, giving us a pearson of 0
    if scoresum > 0.:
        val =  nsum/scoresum + baseline
    else:
        val=baseline
    return val
```




```python
def find_RMSE(df,k,reg):
    uid=df.user_id.values
    bid=df.business_id.values
    actual=df.review_score.values
    predicted=np.zeros(len(actual))
    counter=0
    for user_id, biz_id in zip(uid,bid):
        predicted[counter]=rating(train_data, db, biz_id, user_id, k=k, reg=reg)
        counter=counter+1
    #compare_results(actual, predicted)
    print("RMSE: ",sqrt(mean_squared_error(actual, predicted)))
```


**Great News! So how did we do?**



```python
from math import sqrt
from sklearn.metrics import mean_squared_error
```




```python
find_RMSE(train_data,3,3.)
find_RMSE(test_data,3,3.)
```


    RMSE:  0.9943598895579021



    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    <ipython-input-40-aafdcb5dc0ef> in <module>()
          1 find_RMSE(train_data,3,3.)
    ----> 2 find_RMSE(test_data,3,3.)


    <ipython-input-37-be94852e5281> in find_RMSE(df, k, reg)
          6     counter=0
          7     for user_id, biz_id in zip(uid,bid):
    ----> 8         predicted[counter]=rating(train_data, db, biz_id, user_id, k=k, reg=reg)
          9         counter=counter+1
         10     #compare_results(actual, predicted)


    <ipython-input-36-22b9dcd48cef> in rating(df, dbase, restaurant_id, user_id, k, reg)
          4     nsum=0.
          5     scoresum=0.
    ----> 6     nears=knearest_amongst_userrated(restaurant_id, user_id, df, dbase, k=k, reg=reg)
          7     restaurant_mean=df[df.business_id==restaurant_id].business_average_rating.values[0]
          8     user_mean=users_reviews.user_average_rating.values[0]


    <ipython-input-35-f8070256e1e3> in knearest_amongst_userrated(restaurant_id, user_id, df, dbase, k, reg)
          2     dfuser=df[df.user_id==user_id]
          3     bizsuserhasrated=dfuser.business_id.unique()
    ----> 4     return knearest(restaurant_id, bizsuserhasrated, dbase, k=k, reg=reg)


    <ipython-input-32-e7fc4dccedc9> in knearest(restaurant_id, set_of_restaurants, dbase, k, reg)
          5     for other_rest_id in set_of_restaurants:
          6         if other_rest_id!=restaurant_id:
    ----> 7             sim, nc=dbase.get(restaurant_id, other_rest_id)
          8             ssim=shrunk_sim(sim, nc, reg=reg)
          9             similars.append((other_rest_id, ssim, nc ))


    <ipython-input-25-7292897fbdd7> in get(self, b1, b2)
         34     def get(self, b1, b2):
         35         "returns a tuple of similarity,common_support given two business ids"
    ---> 36         sim=self.database_sim[self.uniquebizids[b1]][self.uniquebizids[b2]]
         37         nsup=self.database_sup[self.uniquebizids[b1]][self.uniquebizids[b2]]
         38         return (sim, nsup)


    KeyError: 'XgUlUmrktr2Um2gczYeYpg'


**What happened?**

Well, we have an error above. As we discovered, we had the same users in both the test and train sets, but we forgot to make sure we had the same restaurants also in test and train. Unfortunately, it took 4 hours to create the similarity database the first time we made it so we thought it best to forego the gray hairs and miss out on seeing our (hopefully low) RMSE for the testing set.

**Good news - we can still do lots of fun things**

We'll take a moment to avoid sulking and instead make some fun predictions. Given some sample business IDs from above, we try to see what recommendations we come up with. Given some test businesses and a test user, let's find out what our recommender would provide.



```python
testbizid="HNs2Nf-trqFTDtho4vhfmA"
testbizid2="SP7H3zPArNvbHKQW0c_gpA"
testuserid="3Uv0dGI2IXJb2OUj8R2GJA"
```




```python
def biznamefromid(df, theid):
    return df['business_name'][df['business_id']==theid].values[0]
def usernamefromid(df, theid):
    return df['user_id'][df['user_id']==theid].values[0]
```


**Given you like business #1, what else might you like?**



```python
tops=knearest(testbizid, train_data.business_id.unique(), db, k=7, reg=3.)
print( "For ",biznamefromid(train_data, testbizid), ", top matches are:")
for i, (biz_id, sim, nc) in enumerate(tops):
    print( i,biznamefromid(train_data,biz_id), "| Sim", sim, "| Support",nc)
```


    For  The South Side , top matches are:
    0 Market Garden Brewery | Sim 0.769230769231 | Support 10
    1 Tremont Taphouse | Sim 0.75 | Support 9
    2 Great Lakes Brewing Company | Sim 0.75 | Support 9
    3 Superior Pho | Sim 0.727272727273 | Support 8
    4 The Greenhouse Tavern | Sim 0.727272727273 | Support 8
    5 Happy Dog | Sim 0.727272727273 | Support 8
    6 Forage Public House | Sim 0.727272727273 | Support 8


**How about business #2?**



```python
tops2=knearest(testbizid2, train_data.business_id.unique(), db, k=7, reg=3.)
print("For ",biznamefromid(train_data, testbizid2), ", top matches are:")
for i, (biz_id, sim, nc) in enumerate(tops2):
    print(i,biznamefromid(train_data,biz_id), "| Sim", sim, "| Support",nc)
```


    For  High Thai'd , top matches are:
    0 Townhall | Sim 0.5 | Support 3
    1 Beachland Ballroom and Tavern | Sim 0.5 | Support 3
    2 Chutney Rolls | Sim 0.5 | Support 3
    3 Felice | Sim 0.5 | Support 3
    4 Tommy's Restaurant | Sim 0.5 | Support 3
    5 Flying Fig | Sim 0.5 | Support 3
    6 Deagan's Kitchen & Bar | Sim 0.5 | Support 3


**And how do we think you'll rate them?**



```python
toprecos=get_top_recos_for_user(testuserid, train_data, db, n=5, k=7, reg=3.)
```




```python
print( "User Average", train_data[train_data.user_id==testuserid].review_score.mean(),"for",usernamefromid(train_data,testuserid))
print( "Predicted ratings for top choices calculated earlier:")
for biz_id,biz_avg in toprecos:
    print( biznamefromid(train_data, biz_id),"|",rating(train_data, db, biz_id, testuserid, k=7, reg=3.),"|","Average",biz_avg)
```


    User Average 3.795918367346939 for 3Uv0dGI2IXJb2OUj8R2GJA
    Predicted ratings for top choices calculated earlier:
    Tremont Taphouse | 3.85087719298 | Average 4.428571428571429
    Great Lakes Brewing Company | 4.13622047244 | Average 4.181818181818182
    Market Garden Brewery | 3.07809530274 | Average 3.5384615384615383
