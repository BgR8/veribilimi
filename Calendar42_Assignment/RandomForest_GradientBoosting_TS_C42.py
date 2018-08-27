
# coding: utf-8

# # Introduction

# In order to solve time series problem indicated in assignment paper I will be using Cross-Industry Process for Data Mining (CRISP-DM) as methodology. CRISP-DM methodology provides a structured approach to planning a data mining project. It is a robust and well-proven methodology. It is composed of six phases:
# 1. Business understanding
# 2. Data understanding
# 3. Data preparation
# 4. Modeling
# 5. Evaluation
# 6. Deployment

# # 1. Business understanding

# Business has time series data about daily worker counts and needs to better understanding this data in order to more accurate worker count planning and predictions.

# # 2-3. Data understanding and preparation

# I choose Python as a programming language. Because it is interpretable, no need to compile, interactive use, datascientist friendly and widely used to solve datascience problems. I use Pandas library for data exploration and preperation.

# In[1]:

# Import libs 
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:

# Read dataset. While reading make date column dataframe index and parse date column as timestamp
dataframe = pd.read_csv("assignment.csv", parse_dates=['date'], index_col="date")


# In[3]:

# First overview of dataset
dataframe.head()


# In[4]:

# Check nulls
dataframe.info()


# No null values. Both date and daily worker count are consistent in terms of row counts.

# In[5]:

# As we see above. The index is not sorted. We should sort it. 
# Sort index and assign dataframe again
dataframe = dataframe.sort_index()


# In[6]:

# Let's see if dataframe index is sorted?
dataframe.head()


# In[7]:

# change column names. We don't want blanks in column names. So we replace underscores.
# create new python list for new column names
column_names = ['daily_worker_count']


# In[8]:

# assign newly created column names list to dataframe columns
dataframe.columns = column_names


# In[9]:

# Check if new column names is assigned
dataframe.columns


# ## Graphical overview to timeseries

# In[10]:


# convert pandas series
dataframe_pd_series = dataframe['daily_worker_count']
# declare the canvas size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 30, 8

# Plot time series
plt.plot(dataframe_pd_series)


# ## Interpretation of graph

# Between 2013 and 2015 work counts are so low but sharp increase in 2015. We can examine the time series into 4 period:
# 1. 2013-2015 period: stable and low variance, no significant trend.
# 2. In 2015 variance increses along with daily work count, slight increasing trend.
# 3. From beginning of 2016 to mid 2017 variance is greater than 2015. We also see highest work counts during this period. No significant trend.
# 4. In the scond half of 2017 variance seems same as previous period but data looses intensity 

# In[11]:

# Extract new features from dataframe index(datetimeindex)
# We create three new features 
dataframe['year'] = dataframe.index.year
dataframe['day_of_month'] = dataframe.index.day
dataframe['month'] = dataframe.index.month
dataframe['day_of_week'] = dataframe.index.dayofweek
dataframe['quarter'] = dataframe.index.quarter
dataframe['week_of_year'] = dataframe.index.weekofyear


# In[12]:

# See if new features has properly created
dataframe.head(20)


# In[13]:

# Check outliers in year, month and day
print("max month:", dataframe.month.max())
print("min month:", dataframe.month.min())
print("max year:", dataframe.year.max())
print("min year:", dataframe.year.min())
print("max day of month:", dataframe.day_of_month.max())
print("min day of month:", dataframe.day_of_month.min())
print("max day of week:", dataframe.day_of_week.max())
print("min day of week:", dataframe.day_of_week.min())


# No outliers in new features

# # Weekday distribution

# In[14]:

# Let's see how data distributed along with day of week
dataframe.groupby('day_of_week').daily_worker_count.agg(['count','min','max','sum','mean'])


# In[15]:

plt.plot(dataframe.groupby('day_of_week').daily_worker_count.mean())
plt.show()


# In[16]:

# As we see above 5 and 6 days have significantly smaller work counts in both mean and sum.
# So we should better create new feature that indicates weekend or not.
is_weekend = []
for i in dataframe.day_of_week:
    if i >= 5:
        is_weekend.append(1)
    else:
        is_weekend.append(0)
        
dataframe['is_weekend'] = is_weekend


# In[17]:

# check is_weekend is added
dataframe.head()


# ### Distribution of work days seems balanced saturday and sunday average worker count is less than work days, and sunday is also less than saturday

# # Monthly distribution

# In[18]:

# Let's see how data distributed along with month of year
dataframe.groupby('month').daily_worker_count.agg(['count','min','max','sum','mean'])


# In[19]:

plt.plot(dataframe.groupby('month').daily_worker_count.mean())
plt.show()


# ### Significant decreasing trend in monthly worker counts

# # Quarterly Distribution

# In[20]:

# Let's see how data distributed along with quarter
dataframe.groupby('quarter').daily_worker_count.agg(['count','min','max','sum','mean'])


# In[21]:

plt.plot(dataframe.groupby('quarter').daily_worker_count.mean())
plt.show()


# # Yearly Distribution

# In[22]:

# Let's see how data distributed along with year
dataframe.groupby('year').daily_worker_count.agg(['count','min','max','sum','mean'])


# In[23]:

plt.plot(dataframe.groupby('year').daily_worker_count.mean())
plt.show()


# ### Sudden increase in 2015. Highest worker counts in 2016 and slight decrease in 2017.

# # 4. Modeling
# ## Building Machine Learning Model

# # 1'st Model:RandomForestRegressor whole data

# In[24]:

# Split columns into features and target values
X = dataframe.iloc[:, 1:].values
y = dataframe.iloc[:, 0].values


# In[25]:

from sklearn.ensemble import RandomForestRegressor
regressor_rf_whole = RandomForestRegressor(n_estimators=100, random_state = 42)
regressor_rf_whole.fit(X,y)


# In[26]:

y_pred_rf_whole = regressor_rf_whole.predict(X)


# First; this is a regression problem.
# Second; since our data is not linear we better use nonlinear regression models.
# 

# In[27]:

from sklearn.metrics import r2_score
r2_score(y, y_pred_rf_whole)


# In[28]:

X.shape


# # 2'nd Model: RandomForestRegressor with train dataset

# r2_score is so high the model seems overfitted, to prevent it let's use split data into train and test.

# In[29]:

# Split dataset into train and test set
from sklearn.model_selection  import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[30]:

from sklearn.ensemble import RandomForestRegressor
regressor_rf = RandomForestRegressor(n_estimators=100, random_state = 0)
regressor_rf.fit(X_train,y_train)


# In[31]:

y_pred_rf = regressor_rf.predict(X_test)


# In[32]:

from sklearn.metrics import r2_score
r2_score(y_test, y_pred_rf)


# In[33]:

X_train.shape


# ## Prediction Function

# In[34]:

'''
# Give the dates of list to be predicted in a csv file. Header must be date.
# Example:

date
01/01/2017
01/02/2016
01/03/2015
01/04/2015
01/05/2017
'''
def featureMarixMaker(will_predicted_csv):
    import pandas as pd
    df = pd.read_csv(will_predicted_csv, parse_dates=['date'], index_col="date")
    df['year'] = df.index.year
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['day_of_week'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['week_of_year'] = df.index.weekofyear
    is_weekend = []
    for i in df.day_of_week:
        if i >= 5:
            is_weekend.append(1)
        else:
            is_weekend.append(0)

    df['is_weekend'] = is_weekend
    X = df.iloc[:, 0:].values
    return X


# In[35]:

# Using featureMarixMaker() function, create a feature matrix
# Don't forget to create csv file including dates to predict in working directory
XX = featureMarixMaker("assignment_pred.csv")

# Prediction
print(regressor_rf.predict(XX))


# In[ ]:




# In[ ]:




# In[ ]:




# # 3'rd Model: GradientBoostingRegressor

# In[36]:

from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

# Prepare the parameters
params = {'n_estimators': 200, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}

# Create GradientBoostingRegressor object with parameters
regressor_gb = ensemble.GradientBoostingRegressor(**params)

# Train GradientBoostingRegressor object with train set
regressor_gb.fit(X_train, y_train)

# Test the model with test set and  compute mean square error (mse)
mse = mean_squared_error(y_test, regressor_gb.predict(X_test))
print("MSE: %.4f" % mse)


# In[37]:

# Make predictions by using trained model. 
y_pred_gb = regressor_gb.predict(X_test)


# In[38]:

r2_score(y_test, y_pred_gb)


# In[39]:

# Using featureMarixMaker() function, create a feature matrix 
# Don't forget to create csv file in including dates to predict in working directory
XX = featureMarixMaker("assignment_pred.csv")

# Prediction
print(regressor_gb.predict(XX))


# In[ ]:



