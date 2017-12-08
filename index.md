---
title: Yelp Recommendations in OH!
---
## By Maia Woluchem, Anthony Sciola, and Jay Dev

<img src="Cleveland_not_Detroit.png" class="img-responsive" alt="From 'Hastily Made Cleveland Tourism Video: 2nd Attempt.'">


## Problem Statement and Motivation

We've heard that building a recommendation system is difficult, so we wanted to give it a try. We are interested in the predictive power of 4 different models in recommending restaurants in Ohio.

## Introduction and Description of the Data

> Jay: Have you ever been to Ohio?

> Anthony: I have, but I don't recommend it.

In this project, we are attempting to do the unwise: make recommendations in favor of Ohio. Using data from reviews in Ohio, available through Yelp's 12th Data Set Challenge data, we have created several recommendation models using linear regression, regularized regression, matrix factorization, and k-nearest neighbors and Pearson's R distance.

As predicted in the project assignment, our recommenders are neither particularly effective nor sophisticated. But through this process, we have learned about handling and modeling large sets of data, alternating least squares and distance-based modeling methods, and the importance of committing changes on GitHub.

The Yelp Dataset Challenge data consisted of six separate data files: check-ins, photos, tips, reviews, users, and businesses. Each of these data sets contain fields relevant to the particular population it describes, either users, establishments, or reviews. We are most interested in the data sets on reviews (which is at the review-level), users (which is at the user-level), and businesses (which is at the establishment-level).

The original reviews data set contains 2,927,859 records. Among the fields provided by Yelp, we were most interested in two: the star rating given in each review (rating) and the date that the review was logged (review_date). We immediately noticed that the mean of rating is 3.70 and the median is 4.00, indicating that the distribution of reviews is skewed towards higher star ratings. The users data contains 1,183,362 user records, with variables describing average rating (average_stars), total number of reviews (review_count), years in which the user had ‘elite’ status (which we aggregate to the number of years with elite status: elite_count), and the date that the user joined Yelp (join_date). In reviewing this user data, we notice that average_stars match the ratings seen in the reviews data set (with mean 3.71 and median 3.89). We can also see that review_count follows an exponential decay function with a long tail—many users post just one or a handful of reviews with a small segment of very active users. This is reflected in the relatively small number users that have ever achieved elite status: less than 5 percent have ever held elite status.

As the Yelp business data set included records beyond the intended scope of our project, we filtered it prior to EDA. According to the project guidelines, we focused our analysis of businesses to those that were categorized as  ‘restaurants.’ We were left with 38,668 business records. While the data set included a large array of features on business characteristics, we found that many of those features had a large share of missing values. As we could not confidently impute these values as False, we determined to minimize the level of missingness among features that we retain for regression analysis. After merging with the reviews data, we removed all variables with more than 50 percent missingness (leaving us with 49 characteristic variables of a possible 93). We were interested in conducting EDA that revealed patterns in ratings based on the cuisine of the establishment, but found 628 unique values for types of cuisine in our dataset. Even when restricting for the most common cuisines, the visualizations were quite cumbersome, so we have forgone including them in this brief. After visualizing the remaining features, we found several which seemed important to include in our regression—particularly location of the restaurant; whether the restaurant takes reservations; is good for breakfast, brunch, lunch, dinner, dessert, and late-night; has delivery; has parking; type of cuisine; and whether the location has Wi-fi.

Within our initial analysis, we looked at the number of check-ins, photos, and tips at each establishment, as well as the number of tips provided by each user, but found that these data were missing for a majority of businesses and users, so we have opted not to ultimately use them.

From our EDA, we came to realize the primary challenge of working with this data: its size. Running most of these procedures on the full data set after merging reviews, users, and businesses seemed unruly, so we decided to cut the data set down. We processed and filtered the data set using Hadoop and Scalding Map Reduce. [ANTHONY WILL FILL IN DETAILS]

After trying a few different data sets, we decided to use the Ohio data for users with at least 150 reviews (except for our matrix factorization model, which uses the national data set for 150 reviews), as it was presented a sufficient number of users and businesses to run our models.

For more on our EDA, please visit the Exploratory Data Analysis page.

## Related Work

We were aided in our analysis of the Yelp data by an analysis of *Collaborative Filtering for Implicit Feedback Datasets*.<sup>1</sup> [ANTHONY WILL TALK ABOUT PAPER HE USED]

We were also aided in understanding the Matrix Factorization model through a paper on *Matrix Factorization Techniques for Recommender Systems*.<sup>2</sup>

Finally, we were aided in creating the Pearson's R distance-based analysis by a published homework notebook from when this course was taught in 2013.<sup>3</sup>

## Modeling Approach and Project Trajectory

You can learn more about our modeling approach as you visit separate pages on our website, which correspond to each model that was run. It is worth noting, however, that our project evolved along with the ability of our laptops to run these data sets through each model. While we started with a data set conceived at the end of the EDA process (that is, one that combined the eight largest states and included all users and businesses), and split across train and test at about a 75-25 rate, this proved impossible to run on our computers through the Regularization Regression, Matrix Factorization, and k-NN distance models. We began trying data sets within each market, and then data sets which cut out users with less than 5 reviews, less than 10 reviews, less than 100 reviews, and so on. Finally, we found our sweet spot to be a mid-sized market (Ohio) with users with at least 150 reviews in the reviews data set.

## Results, Conclusions, and Future Work

As a result of our analysis, we have found that it is really difficult to improve on baseline results. As shown in the regularization and distance-based procedures, the baseline model with average user ratings and average business ratings did as well or slightly better.

A shortcoming of our analysis is the difficulty of comparing results due to the different size constraints and specifications of our data sets. For instance, we were pushed to use global data on the matrix factorization model as it was guaranteed to have all users and restaurants in both train and test data sets (while Ohio had some discrepancies between restaurants in train and test).

If given more time, we would try to incorporate more of the data published by Yelp. We initially were interested in whether market-specific recommendation models were more accurate than global recommendation models, but we were not able to compare them due to processing and time constraints. We would hope to do so with greater computing power and time in the future.  



## References
<sup>1</sup> Hu, Y., Y. Koren, and C. Volinsky. "Collaborative Filtering for Implicit Feedback Datasets." Accessed from http://yifanhu.net/PUB/cf.pdf.

<sup>2</sup> Koren, Y., R. Bell, and C. Volinsky. "Matrix Factorization Techniques for Recommender Systems." Accessed from https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf.

<sup>3</sup> Blitzstein, J., H. Pfister, V. Kaynig-Fittkau, and R. Dave. "HW4: Do we really need Chocolate Recommendations?" Accessed from http://nbviewer.jupyter.org/github/cs109/content/blob/master/HW4_solutions.ipynb.
