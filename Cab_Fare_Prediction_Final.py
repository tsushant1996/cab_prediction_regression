
# coding: utf-8

# # Cab fare prediction

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
from math import radians, cos, sin, asin, sqrt
from sklearn.model_selection import train_test_split
import time
import datetime
import calendar
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')
import random
import pickle
random.seed(113)


from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt


# We have two files one for training and other for test
# <ul>
#  <li>train_cab.csv</li>
#  <li>test_cab.csv</li>
# <ul> 

# In[2]:


df_train = pd.read_csv("train_cab.csv");

df_test = pd.read_csv("test.csv")


# In[3]:


#Let us check the shape of dataset
print(df_train.shape)


# In[4]:


#Let us check some of the data in dataframe
df_train.head()


# In[5]:


#Let us have some information about the dataframe like null values and types
df_train.info()


# In[6]:


#Summing all the null values in all the columns
df_train.isnull().sum()


# There are 55 entries in passenger_count.synthesizing the value for 55 rows in passenger count and 24 entries in fare_amount would not be great idea.So we will remove the these rows from particular dataframe

# In[7]:



"""
on further analysis we have seen in most of the column passenger count in 1 so instead of dropping the rows
we can fill it with median,mean or using model like knn or we can fill with one of the frequent items
"""
# df_train = df_train.dropna(how = 'any', axis = 'rows')

df_train['passenger_count'].value_counts().plot.bar(color = 'b', edgecolor = 'k');
plt.title('Passenger Counts');
plt.xlabel('Number of Passengers'); 
plt.ylabel('Count');
plt.show()


# In[8]:


"""

Above graph shows that passenger_count with value 1 is occurring more frequently so we can fill it with 1
"""

df_train[['passenger_count']] = df_train[['passenger_count']].fillna(value=1)


# In[9]:


df_train.isnull().sum()


# In[10]:


df_train = df_train.dropna(how = 'any', axis = 'rows')


# In[11]:


print(df_train.shape)


# <h5>We will start by doing some analysis on the basis of passenger_count,
# which will help us to remove outlier and corrupted value</h5>

# In[12]:


#As passenger count cannot be float let us convert it in integer
df_train['passenger_count'] = df_train['passenger_count'].astype(int)


# In[13]:


#checking for outliers in box plot or any corrupted data
boxplot = df_train.boxplot(column=['passenger_count'])
plt.show()


# In[14]:


df_train['passenger_count'].value_counts().plot.bar(color = 'b', edgecolor = 'k');
plt.title('Passenger Counts');
plt.xlabel('Number of Passengers'); 
plt.ylabel('Count');
plt.show()


# As we can see in boxplot the range of data is very huge  like 5000 which is unrealistic therefore it implies that we have some corrupted data and also most of our data lies between 1 to 8 as we can see in graph above. 
# <li>In reality maximum the cab can afford is 8 passenger,so we will remove all the column whcih has more than 8 passengers</li>

# In[15]:


df_train = df_train.drop(df_train[df_train['passenger_count'] > 8 ].index,axis=0)


# In[16]:


#Removing rows with 0 passengers as it is of no use
df_train = df_train.drop(df_train[df_train['passenger_count'] <= 0 ].index,axis=0)


# In[17]:


print(df_train.shape)


# <h5>Now we will do preprocessing on the basis of fare_amount

# In[18]:


"""
As in dataframe information we can see fare_amount is in object/string 
so our first task is to convert the fare amount in float value then we 
will start other preprocessing
"""

df_train['fare_amount'] = df_train['fare_amount'].apply(pd.to_numeric, errors='coerce')


# In[19]:


#As we have taken errors as coerce and hence row which will be having errors will be set to NaN
print(df_train.isnull().sum())


# In[20]:


#We will go on removing NaN row
df_train = df_train.dropna(how = 'any', axis = 'rows')


# In[21]:


print(df_train.shape)


# In[22]:


#Also fare_amount cannot be negative or zero and hence remove all the rows having fare_amount as zero or negative
df_train = df_train.drop(df_train[df_train['fare_amount'] <= 0 ].index,axis=0)


# In[23]:


print(df_train.shape)


# In[24]:


#checking for outliers in box plot or any corrupted data
boxplot = df_train.boxplot(column=['fare_amount'])
plt.show()
  


# We can see there are some data which are more than 50000 in boxplot graph.Let us see how many rows we have that has fare_amount more than 5000 and further we will take random fare amount and explore where most of our data lies. as most of the point lies inside 5000
# 

# In[25]:


print(len(df_train[df_train['fare_amount']<5000]))
print(len(df_train[df_train['fare_amount']<1000]))
print(len(df_train[df_train['fare_amount']<500]))
print(len(df_train[df_train['fare_amount']<100]))


# In[26]:


"""
As most of our data lies less than 100 we will remove all the rows fare_amount more than 100 whcih will surely help
our model
"""
df_train = df_train.drop(df_train[df_train['fare_amount'] > 100 ].index,axis=0)


# In[27]:


df_train[df_train['fare_amount']<100].fare_amount.hist(bins=10, figsize=(14,3))
plt.show()


# In[28]:


# we will also remove some values which are not zero but close to zero
df_train[df_train['fare_amount'] < 1 ]


# In[29]:


df_train = df_train.drop(df_train[df_train['fare_amount'] < 1 ].index,axis=0)


# In[30]:


print(df_train.shape)


# In[31]:


df_train.head(20)


# As we can see we also have some values in which all the pickup latitude and lattude  are 0 which means they are corrupted so we will remove all the data with all the 4 attributes with 0 values

# In[32]:


df_train = df_train.drop(df_train[
    (df_train['pickup_longitude'] == 0) 
    & (df_train['pickup_latitude'] == 0) 
    & (df_train['dropoff_longitude'] == 0) 
    & (df_train['dropoff_latitude'] == 0 )
].index, axis=0)
print(df_train.shape)


# In[33]:


#Some of the research has given me a values in which lattitude and longitude lies
#Lattitude Lie between -90 to  90
#Longitude Lie between -180 to 180
df_train[
    (df_train['pickup_longitude'] > 180) 
    |  (df_train['pickup_longitude'] < -180) 
    | (df_train['pickup_latitude'] > 90) 
    | (df_train['pickup_latitude'] < -90) 
    | (df_train['dropoff_longitude'] > 180 )
    | (df_train['dropoff_longitude'] < -180 )
    | (df_train['dropoff_latitude'] > 90) 
    | (df_train['dropoff_latitude'] < -90)
]


# In[34]:


#As the above data is correct we have some weird value for pickup we can remove this data

df_train = df_train.drop(df_train[
    (df_train['pickup_longitude'] > 180) 
    |  (df_train['pickup_longitude'] < -180) 
    | (df_train['pickup_latitude'] > 90) 
    | (df_train['pickup_latitude'] < -90) 
    | (df_train['dropoff_longitude'] > 180 )
    | (df_train['dropoff_longitude'] < -180 )
    | (df_train['dropoff_latitude'] > 90) 
    | (df_train['dropoff_latitude'] < -90)
].index , axis=0)


print(df_train.shape)


# As we have Latitude and longitude there must be some way we can calulate the distance travelled,Found following formulae on calculating distance from latitudes and longitudes
# https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula

# In[35]:


def sphere_dist(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    """
    Return distance along great radius between pickup and dropoff coordinates.
    """
    #Define earth radius (km)
    R_earth = 6371
    #Convert degrees to radians
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(np.radians,
                                                             [pickup_lat, pickup_lon, 
                                                              dropoff_lat, dropoff_lon])
    #Compute distances along lat, lon dimensions
    dlat = dropoff_lat - pickup_lat
    dlon = dropoff_lon - pickup_lon
    
    #Compute haversine distance
    a = np.sin(dlat/2.0)**2 + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon/2.0)**2
    return 2 * R_earth * np.arcsin(np.sqrt(a))


# In[36]:


df_train['haversine_distance'] = df_train.apply(lambda row : sphere_dist(row['pickup_latitude']
                                                           ,row['pickup_longitude'],
                                                           row['dropoff_latitude'],
                                                           row['dropoff_longitude'])
                                                          , axis = 1)


# In[37]:


df_train = df_train.drop(df_train[
    (df_train['haversine_distance'] == 0)].index , axis=0)



# In[38]:


df_train[df_train['haversine_distance'] >100 ]


# In[39]:


#As we can see there are still some values which are huge so we can drop it
df_train = df_train.drop(df_train[
    (df_train['pickup_longitude'] == 0) 
    |  (df_train['pickup_longitude'] == 0) 
    | (df_train['pickup_latitude'] == 0) 
    | (df_train['pickup_latitude'] == 0) 
    | (df_train['dropoff_longitude'] == 0 )
    | (df_train['dropoff_longitude'] ==  0 )
    | (df_train['dropoff_latitude'] == 0) 
    | (df_train['dropoff_latitude'] == 0)
].index, axis= 0)


# In[40]:


df_train[df_train['haversine_distance'] > 50 ].shape


# In[41]:


boxplot = df_train.boxplot(column=['haversine_distance'])
plt.show()

print(len(df_train[df_train['haversine_distance']>5000]))
print(len(df_train[df_train['haversine_distance']>1000]))
print(len(df_train[df_train['haversine_distance']>500]))
print(len(df_train[df_train['haversine_distance'] > 50]))


df_train[df_train['haversine_distance'] > 50].sort_values("haversine_distance",ascending=False)

df_train[df_train['haversine_distance'] > 50].shape

# df_train = df_train.drop(df_train[
#     (df_train['haversine_distance'] > 100)].index , axis=0)
# df_train[df_train['haversine_distance']<100].haversine_distance(bins=10, figsize=(14,3))


# In[42]:


df_train[df_train['haversine_distance'] < 50].sort_values("haversine_distance",ascending=False)


# With some analysis on haversine distance we have found some haversine distance is very huge there is also low fare amount for some haversine ditance whcih are greater than 50 so we canremove all the rows which have haversine distance more than 50 and we have 13 rows like that 
# <li>Fare can be low for high distance because of mistake in data collection</li>
# <li>One of the possibilty that the fare is low is because of some coupons or scheme</li>
# 

# In[43]:


df_train = df_train.drop(df_train[df_train['haversine_distance'] > 50 ].index,axis=0)


# In[44]:



boxplot = df_train.boxplot(column=['haversine_distance'])
plt.show()


# In[45]:


df_train.shape


# In[46]:


df_train.head(10)


# In[47]:


"""
Also converting latitudes and longitudes to radians
"""
def radian_conv(degree):
    """
    Return radian.
    """
    return  np.radians(degree)
df_train['pickup_longitude']  = radian_conv(df_train['pickup_longitude'])
df_train['pickup_latitude']  = radian_conv(df_train['pickup_latitude'])
df_train['dropoff_longitude']  = radian_conv(df_train['dropoff_longitude'])
df_train['dropoff_latitude']  = radian_conv(df_train['dropoff_latitude'])


# In[48]:


f, ax = plt.subplots(figsize=(10, 8))
corr =df_train.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
plt.show()


# In[49]:


df_train.groupby('passenger_count')['fare_amount'].mean().plot.bar(color = 'b');
plt.title('Average Fare by Passenger Count');
plt.show()


# We can see some changes in passenger count which is small so we will select it to feed in model

# In[50]:


# splitting data in to test and train
train, test = train_test_split(df_train, test_size=0.2)
    


# In[51]:


print("train data shape",train.shape)
print("test data shape",test.shape)


# In[52]:


train.head()


# In[53]:


X_train  = train.drop(['fare_amount','pickup_datetime'], axis=1)

Y_train = train[['fare_amount']]


X_test  = test.drop(['fare_amount','pickup_datetime'], axis=1)

Y_test = test[['fare_amount']]


# In[54]:


X_train.head()


# In[55]:


Y_train.head()


# In[56]:


def linear_regression(x_train,y_train,x_test,y_test):
    regressor = LinearRegression()  
    regressor = regressor.fit(x_train, y_train) #training the algorithm
    y_pred = regressor.predict(x_train)
    rmse_train =  sqrt(mean_squared_error(y_train, y_pred))
#     print("rmse for simple linear regression for train data",rmse_train)
    y_pred_test = regressor.predict(x_test)
    rmse_test =  sqrt(mean_squared_error(y_test, y_pred_test))
#     print("rmse for simple linear regression for test data",rmse_test)
    return {'rmse_train':rmse_train,'rmse_test':rmse_test}


# In[57]:


def decision_trees(x_train,y_train,x_test,y_test):
    clf =  DecisionTreeRegressor(random_state = 0)
    clf = clf.fit(x_train, y_train)
    y_pred_tree = clf.predict(x_train)
    rmse_train =  sqrt(mean_squared_error(y_train, y_pred_tree))
#     print("rmse for decision trees for train data",rmse)
    y_pred_tree_test = clf.predict(x_test)
    rmse_test =  sqrt(mean_squared_error(y_test,y_pred_tree_test))
#     print("rmse for decision trees for test data",rmse)
    
    return {'rmse_train':rmse_train,'rmse_test':rmse_test}


# In[58]:



    

def random_forest_model(x_train,y_train,x_test,y_test,depth):
#     print(depth)
    regr = RandomForestRegressor(max_depth=depth, random_state=0, n_estimators=100)
    regr = regr.fit(X_train,Y_train)
    feature_importances = pd.DataFrame({'feature': X_train.columns,
                                        'importance': regr.feature_importances_}).\
                           sort_values('importance', ascending = False).set_index('feature')
                           
    y_pred_tree_random_forest = regr.predict(X_train)
    rmse_train =  sqrt(mean_squared_error(Y_train, y_pred_tree_random_forest))
#     print("rmse of Random trees for train data for dpth {} is {}".format(diff_depths[i],rmse))
    y_pred_tree_test_random = regr.predict(X_test)
    rmse_test =  sqrt(mean_squared_error(Y_test,y_pred_tree_test_random))
#     print("rmse of Random trees for test data for depth {} is {}".format(diff_depths[i],rmse))
#     print(feature_importances.head(10))
    
    return {'rmse_train':rmse_train,'rmse_test':rmse_test}
        
        
    


# In[59]:


linear_regression_rmse = linear_regression(X_train,Y_train,X_test,Y_test)

print("linear_regression",linear_regression_rmse)


# In[60]:


decision_trees_rmse = decision_trees(X_train,Y_train,X_test,Y_test)
print("decison trees",decision_trees_rmse)


# In[61]:


def knn_model(x_train,y_train,x_test,y_test,n):
    neigh = KNeighborsRegressor(n_neighbors=n)
    neigh = neigh.fit(X_train,Y_train)
    y_pred_tree_knn = neigh.predict(X_train)
    rmse_train =  sqrt(mean_squared_error(Y_train,y_pred_tree_knn))
#     print("rmse of knn for train data for {} neighbour is {}".format(diff_neighbours[i],rmse))
    y_pred_knn_test = neigh.predict(X_test)
    rmse_test =  sqrt(mean_squared_error(Y_test,y_pred_knn_test))
    return {'rmse_train':rmse_train,'rmse_test':rmse_test}
#     print("rmse of knn for test data for {} neighbour is {}".format(diff_neighbours[i],rmse))

    
    
def knn(x_train,y_train,x_test,y_test):    
    diff_neighbours = [1,2,3,4,5,6,7,8,9]
    for i in range(0,len(diff_neighbours)):
        rmse_errors = knn_model(x_train,y_train,x_test,y_test,diff_neighbours[i])
        
        print("rmse of knn  for train data for dpth {} is {}".format(diff_neighbours[i],rmse_errors['rmse_train']))
        print("rmse of knn  for test data for dpth {} is {}".format(diff_neighbours[i],rmse_errors['rmse_test']))
        
    


# In[62]:


knn(X_train,Y_train,X_test,Y_test)


# In[63]:


def random_forest(x_train,y_train,x_test,y_test):
    diff_depths = [1,2,3,4,5,6,7,8,9,10,11,12,13]
    
    for i in range(0,len(diff_depths)):
#         print(diff_depths[i])   
        rmse_errors = random_forest_model(x_train,y_train,x_test,y_test,diff_depths[i])
        
        print("rmse of Random trees for train data for dpth {} is {}".format(diff_depths[i],rmse_errors['rmse_train']))
        print("rmse of Random trees for test data for dpth {} is {}".format(diff_depths[i],rmse_errors['rmse_test']))


    return

random_forest(X_train,Y_train,X_test,Y_test)


# 
# <h3>We can take few insights from these regressions</h3>
# <ul>
# <li>1.Simple Linear regression has performed well but not so well that we can use this in prediction.</li>
# <li>2.RMSE for decision trees for test data is higher than that of test data and hence we can say our decision tree has overfit.</li>
# <li>3.Random forest has performed very well on this data and its rmse for train and test both are low and hence
# we can use Random forest in production for this model.</li>
# <li>4.KNN has performed also well on this data but not as good as random forest and hence we will freeze random forest to be use for production.</li>
# <ul>

# Let us do some feature engineering by adding some new features
# and after that we will look at our model performance
# 

# As we know cab fare besides from passenger and distance also depends upon some factors like weekends,festival season,pickup hour or more let us use the pickup datetime feature so we could explore more about that
# 
# <li>People more likely to go office on weekdays rather on weekends</li>
# <li>Most of the people in real go to office in morning rather in night</li>

# In[64]:


#Formatting the picup daytime so we could split it in hour, day, week and year
df_train['pickup_datetime']=pd.to_datetime(df_train['pickup_datetime'],format='%Y-%m-%d %H:%M:%S UTC', errors='coerce')


# In[65]:


df_train.isnull().sum()


# In[66]:


df_train = df_train.dropna(how = 'any', axis = 'rows')


# In[67]:


df_train.isnull().sum()


# In[68]:


#We make columns for day,hour ,week,year
df_train['pickup_date']= df_train['pickup_datetime'].dt.date
df_train['pickup_day']=df_train['pickup_datetime'].apply(lambda x:x.day)
df_train['pickup_hour']=df_train['pickup_datetime'].apply(lambda x:x.hour)
df_train['pickup_day_of_week']=df_train['pickup_datetime'].apply(lambda x:calendar.day_name[x.weekday()])
df_train['pickup_month']=df_train['pickup_datetime'].apply(lambda x:x.month)
df_train['pickup_year']=df_train['pickup_datetime'].apply(lambda x:x.year)


# In[69]:


df_train.head()


# In[70]:


f, ax = plt.subplots(figsize=(10, 8))
corr =df_train.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
plt.show()


# In[71]:


df_train.groupby('passenger_count')['fare_amount'].mean().plot.bar(color = 'b');
plt.title('Average Fare by Passenger Count');
plt.show()


# In[72]:


df_train.groupby('pickup_year')['fare_amount'].mean().plot.bar(color = 'b');
plt.title('Average Fare by Year');
plt.show()


# In[73]:


df_train.groupby('pickup_day_of_week')['fare_amount'].mean().plot.bar(color = 'b');
plt.title('Average Fare by week');
plt.show()



# In[74]:


plt.bar(df_train['pickup_day_of_week'], df_train['fare_amount'],  width = 0.8) 

# naming the x-axis 
plt.xlabel('pickup_day_of_week') 
# naming the y-axis 
plt.ylabel('fare_amount') 
# plot title 
plt.title('Bar graph!')
plt.show()


# In[75]:


df_train.groupby('pickup_hour')['fare_amount'].mean().plot.bar(color = 'b');
plt.title('Average Fare by hour');
plt.show()


# In[76]:


df_train.groupby('pickup_month')['fare_amount'].mean().plot.bar(color = 'b');
plt.title('Average Fare by month');
plt.show()


# In[77]:


corrs = df_train.corr()
corrs['fare_amount'].plot.bar(color = 'b');
plt.title('Correlation with Fare Amount');
plt.show()


# From above analysis we can observe that
# 
# <li>Fare amount increases every year so it one of the variable whcih can help our model to  perform better</li>
# <li>Far amount in some pickup hour is more as compare to others so it could also help our model to perform better</li>
# <li>Fare amount is greater on some days as compared to others</li>
# 

# In[78]:


#Let us first convert the pickup_day_of_week in to numerical variable as it is categorical variable
# cat_columns = ['pickup_day_of_week']
# df_train = pd.get_dummies(df_train, prefix_sep="__",
#                               columns=cat_columns)
# df_train = pd.get_dummies(df_train, prefix_sep="__",
#                               columns=['pickup_month'])
# df_train = pd.get_dummies(df_train, prefix_sep="__",
#                               columns=['pickup_hour'])


# In[79]:


df_train.head()


# In[80]:


"""
We can split our data in 80:20
where 80% data will be our training data and 20% data will be our test data
"""
train, test = train_test_split(df_train, test_size=0.2)


# In[81]:


train.shape


# In[82]:


test.shape


# In[83]:


X_train  = train.drop(['fare_amount','pickup_datetime','pickup_date','pickup_day','pickup_day_of_week'
                      ], axis=1)


# In[84]:


Y_train = train[['fare_amount']]


# In[85]:


X_train.head()


# In[86]:


Y_train.head()


# In[87]:


X_test  = test.drop(['fare_amount','pickup_datetime','pickup_date','pickup_day','pickup_day_of_week'
                      ] ,axis=1)
Y_test = test[['fare_amount']]


# In[88]:


Y_train.shape


# In[89]:


linear_regression_rmse = linear_regression(X_train,Y_train,X_test,Y_test)

print("linear_regression",linear_regression_rmse)


# In[90]:


decision_trees_rmse = decision_trees(X_train,Y_train,X_test,Y_test)
print("decison trees",decision_trees_rmse)


# In[91]:


# def random_forest(x_train,y_train,x_test,y_test):
#     diff_depths = [1,2,3,4,5,6,7,8,9,10,11,12,13]
    
#     for i in range(0,len(diff_depths)):
# #         print(diff_depths[i])   
#         rmse_errors = random_forest_model(x_train,y_train,x_test,y_test,diff_depths[i])
        
#         print("rmse of Random trees for train data for dpth {} is {}".format(diff_depths[i],rmse_errors['rmse_train']))
#         print("rmse of Random trees for test data for dpth {} is {}".format(diff_depths[i],rmse_errors['rmse_test']))


#     return

random_forest(X_train,Y_train,X_test,Y_test)


# In[92]:


# def knn(x_train,y_train,x_test,y_test)    
#     diff_neighbours = [1,2,3,4,5,6,7,8,9]
#     for i in range(0,len(diff_neighbours)):
#         rmse_errors = knn_model(x_train,y_train,x_test,y_test,diff_neighbours[i])
        
#         print("rmse of Decision Trr for train data for dpth {} is {}".format(diff_neighbours[i],rmse_errors['rmse_train']))
#         print("rmse of Decision Tree for test data for dpth {} is {}".format(diff_neighbours[i],rmse_errors['rmse_test']))

        
        
knn(X_train,Y_train,X_test,Y_test)        


# 
# <h3>We can take few insights from these regressions</h3>
# <ul>
# <li>1.Simple Linear regression has performed well but not so well that we can use this in prediction.</li>
# <li>2.RMSE for decision trees for test data is higher than that of test data and hence we can say our decision tree has overfit.</li>
# <li>3.Random forest has performed very well on this data and its rmse for train and test both are low and hence
# we can use Random forest in production for this model.</li>
# <li>4.KNN has performed also well on this data but not as good as random forest and hence we will freeze random forest to be use for production.</li>
# <ul>

# #Now we will use real test data and predict the fare_amount,

# In[93]:


df_test = pd.read_csv('test.csv')

df_test.head()


# In[94]:


df_test.info()


# In[95]:


"""
Columns in test dataset don't have missing value and also all the variables are in right format 
so we can directly go on to some advance preprocesing
"""

"""
Writing generic functions that can be reused
"""

#Function for preprocessing based on passenger data

def preprocess_test_dataset(df_dataset_test):
    df_dataset_test = df_dataset_test.drop(df_dataset_test[df_dataset_test['passenger_count'] > 10 ].index,axis=0)
    
    df_dataset_test = df_dataset_test.drop(df_dataset_test[
        (df_dataset_test['pickup_longitude'] == 0) 
        & (df_dataset_test['pickup_latitude'] == 0) 
        & (df_dataset_test['dropoff_longitude'] == 0) 
        & (df_dataset_test['dropoff_latitude'] == 0 )
    ].index, axis=0)
    
    df_dataset_test = df_dataset_test.drop(df_dataset_test[
    (df_dataset_test['pickup_longitude'] > 180) 
    |  (df_dataset_test['pickup_longitude'] < -180) 
    | (df_dataset_test['pickup_latitude'] > 90) 
    | (df_dataset_test['pickup_latitude'] < -90) 
    | (df_dataset_test['dropoff_longitude'] > 180 )
    | (df_dataset_test['dropoff_longitude'] < -180 )
    | (df_dataset_test['dropoff_latitude'] > 90) 
    | (df_dataset_test['dropoff_latitude'] < -90)
].index , axis=0)
    
    df_dataset_test['haversine_distance'] = df_dataset_test.apply(lambda row : sphere_dist(row['pickup_latitude']
                                                            ,row['pickup_longitude'],
                                                            row['dropoff_latitude'],
                                                            row['dropoff_longitude'])
                                                           , axis = 1)
    df_dataset_test['pickup_datetime']=pd.to_datetime(df_dataset_test['pickup_datetime'],
                                                      format='%Y-%m-%d %H:%M:%S UTC', errors='coerce')
    
    df_dataset_test = df_dataset_test.dropna(how = 'any', axis = 'rows')
    
    df_dataset_test['pickup_date']= df_dataset_test['pickup_datetime'].dt.date
    df_dataset_test['pickup_day']=df_dataset_test['pickup_datetime'].apply(lambda x:x.day)
    df_dataset_test['pickup_hour']=df_dataset_test['pickup_datetime'].apply(lambda x:x.hour)
    df_dataset_test['pickup_day_of_week']=df_dataset_test['pickup_datetime'].apply(lambda x:calendar.day_name[x.weekday()])
    df_dataset_test['pickup_month']=df_dataset_test['pickup_datetime'].apply(lambda x:x.month)
    df_dataset_test['pickup_year']=df_dataset_test['pickup_datetime'].apply(lambda x:x.year)
    
    
    
#     df_dataset_test['is_weekend'] = df_dataset_test['pickup_day_of_week'].apply(isWeekend)
    
#     cat_columns = ['pickup_day_of_week']
#     df_dataset_test = pd.get_dummies(df_dataset_test, prefix_sep="__",
#                                   columns=cat_columns)
#     df_dataset_test = pd.get_dummies(df_dataset_test, prefix_sep="__",
#                               columns=['pickup_month'])
    
    df_dataset_test['pickup_longitude']  = radian_conv(df_dataset_test['pickup_longitude'])
    df_dataset_test['pickup_latitude']  = radian_conv(df_dataset_test['pickup_latitude'])
    df_dataset_test['dropoff_longitude']  = radian_conv(df_dataset_test['dropoff_longitude'])
    df_dataset_test['dropoff_latitude']  = radian_conv(df_dataset_test['dropoff_latitude'])

    colnames_drop = ['pickup_date','pickup_day','pickup_datetime','pickup_day_of_week']
    df_dataset_test = df_dataset_test.drop(colnames_drop, axis=1)
    
    
    
    return df_dataset_test
    
    
    
    
    
    
    
    
    



# In[96]:


df_test = preprocess_test_dataset(df_test)


# In[97]:


df_test.shape


# In[98]:


df_test.head()


# In[99]:


df_test.info()


# In[100]:


"""
for experimentation purpose we have build our random forest model on 
diff max_depth, as there is no much difference 
after max-depth 4 we can choose max depth as 4 and n_estmators as 100 and 
we can again train the model for prediction.
"""

regr = RandomForestRegressor(max_depth=7, random_state=0, n_estimators=100)
regr = regr.fit(X_train,Y_train)

save_rf_model = pickle.dumps(regr) 
  
# Load the pickled model 
regr_from_pickle = pickle.loads(save_rf_model) 
  
# Use the loaded pickled model to make predictions 

df_test['fare_amount'] = regr_from_pickle.predict(df_test)



df_test.head()    


# In[101]:


df_test.to_csv(r'test_result.csv',index=False)
df_result = pd.read_csv('test_result.csv')
df_result.head()

