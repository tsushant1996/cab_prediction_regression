rm(list = ls())

#install.packages("tidyr",repos = "http://cran.us.r-project.org")
#install.packages("ggplot2",repos = "http://cran.us.r-project.org")
#install.packages("ggpubr",repos = "http://cran.us.r-project.org")
#install.packages("NISTunits", dependencies = TRUE,repos = "http://cran.us.r-project.org")
#install.packages("corrplot",repos = "http://cran.us.r-project.org")
#install.packages("Hmisc",repos = "http://cran.us.r-project.org")
#install.packages("dplyr",repos = "http://cran.us.r-project.org")
#install.packages("ISLR",repos = "http://cran.us.r-project.org")
#install.packages("rpart",repos = "http://cran.us.r-project.org")
#install.packages("randomForest",repos = "http://cran.us.r-project.org")
#install.packages("lubridate",repos = "http://cran.us.r-project.org")
#install.packages("caret",repos = "http://cran.us.r-project.org")
#install.packages("mlbench",repos = "http://cran.us.r-project.org")





library(lubridate)
library(rpart)
library(randomForest)
library(mlbench)
library(caret)
library("ISLR")
library("Hmisc")
library("dplyr")
library("tidyr")
library("ggplot2")
library(NISTunits)
library(corrplot)




#changing directory for working
setwd("/home/sushant/machine_learning_cab_project/")

#importing both train and test data
df_train = read.csv("train_cab.csv", header = T, as.is = T)

df_test = read.csv("test.csv", header = T, as.is = T)




set.seed(0)
#---------------------------

cat("Earth Radius...\n")
R_earth <- 6371

print(dim(df_train))

#dataframe consist of 16067 rows and 7 columns

print(summary(df_train))

barplot(table(df_train$passenger_count))

#Most of passenger_count has value 1 so we will fill the na data with value 1

df_train$passenger_count = df_train$passenger_count %>% replace_na(1)


# converting fare amount column in double 

df_train$fare_amount = as.double(df_train$fare_amount)


boxplot(df_train$fare_amount)

"
As we have only 25 column fare amount as we 
have huge range in fare amount so we will remove the rows
"
df_train <- df_train %>% drop_na()



print(summary(df_train))


"converting passenger_count in integer as it cannot be in decimals
"
 
df_train$passenger_count = as.integer(df_train$passenger_count)

boxplot(df_train$passenger_count)

"As we can see in boxplot the range of data is very huge like 5000 which is unrealistic
therefore it implies that we have some corrupted data and also most of our
data lies between 1 to 8 as we can see in graph above.
In reality maximum the cab can afford is 8 passenger,so we will remove all the column which
has more than 8 passengers"

"Removing all the rows whoch contains passenger more than 8"
df_train<-df_train[!(df_train$passenger_count > 8),]

boxplot(df_train$passenger_count)

"Removing all the rows whcih contains passenger_count less than 0 or equal to 0 "
df_train<-df_train[!(df_train$passenger_count <= 0),]


"Removing all the Fare amount which is less than or negative as we know fare mount cannot be negative"
df_train<-df_train[!(df_train$fare_amount <= 0),]

boxplot(df_train$fare_amount)

"There are some points whcih are huge in numbers like some trips havae fare_amount more 
than 50000 it can be outlier so let us do the analysis and after that we will remove it"

print(length(df_train[!(df_train$fare_amount > 5000),]))

print(length(df_train[!(df_train$fare_amount > 1000),]))

print(length(df_train[!(df_train$fare_amount > 500),]))

print(length(df_train[!(df_train$fare_amount > 100),]))



df_train<-df_train[!(df_train$fare_amount > 100),]


"we will also like to drop some rows which have fare_amount close to zero"

print(df_train[(df_train$fare_amount <= 1),])

"we have one row with fare amount almost close to zero so we will remove it"

df_train<-df_train[!(df_train$fare_amount <= 1),]

hist(df_train$fare_amount[(df_train$fare_amount < 100)])



"As we can see most of our point lies in 1 to 100 only"

"If we will observe we can see we have some rows 
whose pick_latitide,longitude and dropoff_latitude are 0"

print(df_train[(df_train$pickup_latitude == 0 
                & df_train$pickup_longitude == 0
                &df_train$dropoff_latitude == 0
                &df_train$dropoff_longitude == 0
                ),])

"Let us remove these columns as they have corrupted data"

df_train = df_train[!(df_train$pickup_latitude == 0 
                     & df_train$pickup_longitude == 0
                     &df_train$dropoff_latitude == 0
                     &df_train$dropoff_longitude == 0
),]


#Some of the research has given me a values in which lattitude and longitude lies
#Lattitude Lie between -90 to  90
#Longitude Lie between -180 to 180

df_train = df_train[!((df_train$pickup_latitude < -90) 
                      | (df_train$pickup_latitude > 90)
                      |  (df_train$pickup_longitude < -180) 
                      | (df_train$pickup_longitude > 180)
                      | (df_train$dropoff_latitude < -90) 
                      | (df_train$dropoff_latitude > 90)
                      | (df_train$dropoff_longitude < -180) 
                      | (df_train$dropoff_longitude > 180)
),]

df_train<-df_train[!((df_train$pickup_longitude ==  0) 
                     |(df_train$pickup_latitude ==  0) 
                     |(df_train$dropoff_longitude ==  0) 
                     |(df_train$dropoff_latitude ==  0) 
),]




calculate_haversine_distance <- function(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
{
  #Compute distances along lat, lon dimensions
  
  #first converting the latitude and longitude in radians
  pickup_lat = NISTdegTOradian(pickup_lat)
  pickup_lon = NISTdegTOradian(pickup_lon)
  dropoff_lat = NISTdegTOradian(dropoff_lat)
  dropoff_lon = NISTdegTOradian(dropoff_lon)
  dlat = dropoff_lat - pickup_lat
  dlon = dropoff_lon - pickup_lon
  
  #Compute  distance
  a = sin(dlat/2.0)**2 + cos(pickup_lat) * cos(dropoff_lat) * sin(dlon/2.0)**2
  
  return (2 * R_earth * asin(sqrt(a)))
  
} 

#calculating trip distance when we have latitude and longitude
df_train$haversine_distance <- calculate_haversine_distance(df_train$pickup_latitude,
                                                 df_train$pickup_longitude,
                                                 df_train$dropoff_latitude,
                                                 df_train$dropoff_longitude)


#Also we have observe that in our dataframe we have some values which are
#with huge haversine distance  whcih have lat or long a szero whcih means on equator or on poles whcih is not possible
df_train<-df_train[!(df_train$haversine_distance <= 0),]

#removing all the rows having haversine distance greater than 50 as they are rare and very huge
df_train<-df_train[!(df_train$haversine_distance > 50),]



df_train$pickup_longitude = NISTdegTOradian(df_train$pickup_longitude)
df_train$pickup_latitude = NISTdegTOradian(df_train$pickup_latitude)
df_train$dropoff_longitude = NISTdegTOradian(df_train$dropoff_longitude)
df_train$dropoff_latitude = NISTdegTOradian(df_train$dropoff_latitude)


df_train_without_pickup_datetime <- select(df_train,-c(pickup_datetime))



M<-cor(df_train_without_pickup_datetime)
head(round(M,2))
corrplot(M, method="color")










"
split data for train and test

"
## 80% of the sample size


smp_size <- floor(0.80 * nrow(df_train))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(df_train)), size = smp_size)

x_train <- df_train[train_ind, ]
x_test <- df_train[-train_ind, ]



"
XXXXXXXXXXXXXXXXXXXXXXXXXXX Linear Regression XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
"
model <- fare_amount ~ passenger_count + pickup_latitude + pickup_longitude + dropoff_latitude + dropoff_longitude + haversine_distance
fit <- lm(model, x_train)
"
x_test <- select(x_test,-c(fare_amount))
"

x_test$predicted_fare <- predict(fit,x_test)




postResample(pred = x_test$predicted_fare, obs = x_test$fare_amount)

"
XXXXXXXXXXXXXXXXXXXXXXXXXXX Decision Tree XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
"

decision_fit<-rpart(model, x_train)

x_test$predicted_fare <- predict(decision_fit,x_test)

postResample(pred = x_test$predicted_fare, obs = x_test$fare_amount)



"
XXXXXXXXXXXXXXXXXXXXXXXXXX Random Forest XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


"
RF_model <- randomForest(model, x_train, importance = TRUE, ntree = 100)

x_test$predicted_fare <- predict(RF_model,x_test)

postResample(pred = x_test$predicted_fare, obs = x_test$fare_amount)


"
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
"




"
As we can see random forest has done  well on test dataset No we will add more feature 
to optimise our model.
We can see in our dataset we have date and time so what we can do we can split the datetime 
to  day month year and week and look for the pattern so that we can optimise our model
"


df_train<-df_train %>%
  mutate(
    pickup_datetime = ymd_hms(pickup_datetime),
    year = as.factor(year(pickup_datetime)),
    month = as.factor(month(pickup_datetime)),
    day = as.numeric(day(pickup_datetime)),
    dayofweek = as.factor(wday(pickup_datetime)),
    hour = as.numeric(hour(pickup_datetime)),
  )


df_train_without_pickup_datetime <- df_train %>% drop_na()


df_train_without_pickup_datetime <- df_train_without_pickup_datetime %>% drop_na()
summary(df_train_without_pickup_datetime)
str(df_train_without_pickup_datetime)
"
Converting factor in  to integer
"
df_train_without_pickup_datetime$year = as.integer(df_train_without_pickup_datetime$year)

df_train_without_pickup_datetime$month = as.integer(df_train_without_pickup_datetime$month)

df_train_without_pickup_datetime$dayofweek = as.integer(df_train_without_pickup_datetime$dayofweek)

str(df_train_without_pickup_datetime)



str(df_train_without_pickup_datetime)
df_train_without_pickup_datetime <- select(df_train_without_pickup_datetime,-c(pickup_datetime))
df_train_without_pickup_datetime <- select(df_train_without_pickup_datetime,-c(dayofweek))

df_train_without_pickup_datetime <- select(df_train_without_pickup_datetime,-c(day))

M<-cor(df_train_without_pickup_datetime)
barplot(M)
head(round(M,2))
corrplot(M, method="color")




df_train_without_pickup_datetime <- df_train_without_pickup_datetime %>% drop_na()

"
Trying model with more features 

"




"
split data for train and test

"
## 80% of the sample size


smp_size <- floor(0.80 * nrow(df_train_without_pickup_datetime))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(df_train_without_pickup_datetime)), size = smp_size)

x_train <- df_train_without_pickup_datetime[train_ind, ]
x_test <- df_train_without_pickup_datetime[-train_ind, ]



"
XXXXXXXXXXXXXXXXXXXXXXXXXXX Linear Regression XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
"
str(df_train_without_pickup_datetime)
model <- fare_amount ~ passenger_count + pickup_latitude + pickup_longitude + dropoff_latitude + dropoff_longitude +  haversine_distance +year + hour + month
fit <- lm(model, x_train)
print(model)
"
x_test <- select(x_test,-c(fare_amount))
"

x_test$predicted_fare <- predict(fit,x_test)




postResample(pred = x_test$predicted_fare, obs = x_test$fare_amount)

"
XXXXXXXXXXXXXXXXXXXXXXXXXXX Decision Tree XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
"

decision_fit<-rpart(model, x_train)

x_test$predicted_fare <- predict(decision_fit,x_test)

postResample(pred = x_test$predicted_fare, obs = x_test$fare_amount)



"
XXXXXXXXXXXXXXXXXXXXXXXXXX Random Forest XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


"
RF_model <- randomForest(model, x_train, importance = TRUE, ntree = 100)

x_test$predicted_fare <- predict(RF_model,x_test)

postResample(pred = x_test$predicted_fare, obs = x_test$fare_amount)




"

As we can see RMSE error is much lower After optimising the model so 
we will predict the fare amount on the basis of this model that we have build above.
Performance of random forest as great so we will be using random forest for prediction

Let us try our model with actual test data and predict the fare_amount
"


"

First of first we will have to preprocess the data so we can use our model to predict the 
fare amount 

"


df_test<-df_test %>%
  mutate(
    pickup_datetime = ymd_hms(pickup_datetime),
    year = as.factor(year(pickup_datetime)),
    month = as.factor(month(pickup_datetime)),
    day = as.numeric(day(pickup_datetime)),
    dayofweek = as.factor(wday(pickup_datetime)),
    hour = as.numeric(hour(pickup_datetime)),
  )

summary(df_test)

str(df_test)


df_test$year = as.integer(df_test$year)

df_test$month = as.integer(df_test$month)

df_test$dayofweek = as.integer(df_test$dayofweek)

df_test <- select(df_test,-c(pickup_datetime))


#calculating trip distance when we have latitude and longitude
df_test$haversine_distance <- calculate_haversine_distance(df_test$pickup_latitude,
                                                            df_test$pickup_longitude,
                                                            df_test$dropoff_latitude,
                                                            df_test$dropoff_longitude)
df_test$pickup_longitude = NISTdegTOradian(df_test$pickup_longitude)
df_test$pickup_latitude = NISTdegTOradian(df_test$pickup_latitude)
df_test$dropoff_longitude = NISTdegTOradian(df_test$dropoff_longitude)
df_test$dropoff_latitude = NISTdegTOradian(df_test$dropoff_latitude)

df_test<-df_test[!(df_test$haversine_distance <= 0),]

df_test <- select(df_test,-c(day))
df_test <- select(df_test,-c(dayofweek))


df_test$fare_amount <- predict(RF_model,df_test)

write.csv(df_test, file = "df_test_result_r.csv")













