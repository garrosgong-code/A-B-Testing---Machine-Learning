#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import sklearn.utils
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)


# ## Data Loading

# In[2]:


control_data = pd.read_csv('data/control_data.csv')
experiment_data = pd.read_csv('data/experiment_data.csv')


# In[3]:


control_data.head()


# In[4]:


experiment_data.head()


# ## Exploratory Data Analysis

# In[5]:


control_data.info()


# In[6]:


experiment_data.info()


# In[7]:


control_data.isna().sum()


# In[8]:


sns.heatmap(control_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[9]:


experiment_data.isna().sum()


# In[10]:


sns.heatmap(experiment_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[11]:


control_data[control_data['Enrollments'].isna()]


# ## Data Wrangling

# In[12]:


# Combine with Experiment data
data_total = pd.concat([control_data, experiment_data])
data_total.sample(10)


# In[13]:


np.random.seed(7)

# Add row id
data_total['row_id'] = data_total.index

# Create a Day of Week feature
data_total['DOW'] = data_total['Date'].str.slice(start=0, stop=3)

# Remove missing data
data_total.dropna(inplace=True)

# Add a binary column Experiment to denote
# if the data was part of the experiment or not (Random)
data_total['Experiment'] = np.random.randint(2, size=len(data_total))

# Remove missing data
data_total.dropna(inplace=True)

# Remove Date and Payments columns
del data_total['Date'], data_total['Payments']

# Shuffle the data(to do random permutations of the collections)
data_total = sklearn.utils.shuffle(data_total)


# In[14]:


# Check the new data
data_total.head()


# In[15]:


# Reorder the columns
data_total = data_total[['row_id', 'Experiment', 'Pageviews', 'Clicks', 'DOW', 'Enrollments']]
data_total.head()


# In[16]:


# Splitting the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_total.loc[:, data_total.columns != 'Enrollments'],
                                                    data_total['Enrollments'], test_size=0.2)


# In[17]:


# Converting strings to numbers
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()
X_train['DOW'] = lb.fit_transform(X_train['DOW'])
X_test['DOW'] = lb.transform(X_test['DOW'])


# In[18]:


X_train.head()


# In[19]:


X_test.head()


# ## Helper functions
# - Function for printing the evaluation scores related to a _regression_ problem
# - Function for plotting the original values and values predicted by the model

# In[20]:


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def calculate_metrics(y_test, y_preds):
    rmse = np.sqrt(mean_squared_error(y_test, y_preds))
    r_sq = r2_score(y_test, y_preds)
    mae = mean_absolute_error(y_test, y_preds)

    print('RMSE Score: {}'.format(rmse))
    print('R2_Squared: {}'.format(r_sq))
    print('MAE Score: {}'.format(mae))


# In[21]:


plt.style.use('ggplot')
def plot_preds(y_test, y_preds, model_name):
    N = len(y_test)
    plt.figure(figsize=(10,5))
    original = plt.scatter(np.arange(1, N+1), y_test, c='blue')
    prediction = plt.scatter(np.arange(1, N+1), y_preds, c='red')
    plt.xticks(np.arange(1, N+1))
    plt.xlabel('# Oberservation')
    plt.ylabel('Enrollments')
    title = 'True labels vs. Predicted Labels ({})'.format(model_name)
    plt.title(title)
    plt.legend((original, prediction), ('Original', 'Prediction'))
    plt.show()


# ## Model 01: Linear regression: A baseline

# In[22]:


import statsmodels.api as sm

X_train_refined = X_train.drop(columns=['row_id'], axis=1)
linear_regression = sm.OLS(y_train, X_train_refined)
linear_regression = linear_regression.fit()


# In[23]:


X_test_refined = X_test.drop(columns=['row_id'], axis=1)
y_preds = linear_regression.predict(X_test_refined)


# In[24]:


calculate_metrics(y_test, y_preds)


# In[25]:


plot_preds(y_test, y_preds, 'Linear Regression')


# In[26]:


print(linear_regression.summary())


# In[27]:


pd.DataFrame(linear_regression.pvalues)    .reset_index()    .rename(columns={'index':'Terms', 0:'p_value'})    .sort_values('p_value')


# ## Model 02: Decision Tree

# In[28]:


from sklearn.tree import DecisionTreeRegressor

dtree = DecisionTreeRegressor(max_depth=5, min_samples_leaf =4, random_state=7)
dtree.fit(X_train_refined, y_train)
y_preds = dtree.predict(X_test_refined)

calculate_metrics(y_test, y_preds)


# In[29]:


plot_preds(y_test, y_preds, 'Decision Tree')


# ## Model 03: `XGBoost`

# In[30]:


from xgboost import XGBRegressor


# In[31]:


X_train['Enrollments'] = y_train
X_test['Enrollments'] = y_test


# In[32]:


xgbr = XGBRegressor(learning_rate=0.01,
                    n_estimators=6000,
                    max_depth=4,
                    min_child_weight=0,
                    gamma=0.6,
                    subsample=0.7,
                    colsample_bytree=0.7,
                    objective='reg:linear',
                    nthread=-1,
                    scale_pos_weight=1,
                    seed=27,
                    reg_alpha=0.00006,
                    random_state=42,
                    verbosity = 0)

xg_reg = xgbr.fit(X_train,y_train)
y_preds = xg_reg.predict(X_test)


# In[33]:


calculate_metrics(y_test, y_preds)


# In[34]:


plot_preds(y_test, y_preds, 'XGBoost')


# ## Model 04: Catboost

# In[35]:


from catboost import CatBoostRegressor


# In[36]:


X_train['Enrollments'] = y_train
X_test['Enrollments'] = y_test


# In[37]:


cb = CatBoostRegressor(silent = True)
model_cb = cb.fit(X_train,y_train)
y_preds = model_cb.predict(X_test)


# In[38]:


calculate_metrics(y_test, y_preds)


# In[39]:


plot_preds(y_test, y_preds, 'XGBoost')


# > **XGBoost Wins!**
