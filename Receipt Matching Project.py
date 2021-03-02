#!/usr/bin/env python
# coding: utf-8

# #### Created By : Sai Madhav Tikkani
# #### Dated : 1st March 2021
# #### Objective : 
#     1. To identify whether transaction-receipt matching vector was able to correctly predit the transaction
#     2. Create a best ML model  which can be used to order a set of transactions by likelihood of matching a receipt image 
#    
# #### Methodology :
#     1. Looking for Missing Values
#     2. Looking for Outliers
#     3. Create target variable for the model
#     4. One hot encoding
#     5. Balance Imbalance Class and treatment
#     6. Trying out a Linear model in Logistic Regression
#     7. Trying out a decision tree model in Random Forest
#     8. Trying out a Non Linear model in XG Boost
#     9. Parameter tuning
#     10. Variable Importance
# #### Suggesting Next Steps : As all ideas can't be tried in limited time
# #### Suggesting Required : Suggesting what would be deployment strategy for the model
# 

# In[1]:


### Import all the required packages to be used for furthar analysis
import pandas as pd
import os
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from IPython.display import display


# In[2]:


### Change the locations for current working directory
os.getcwd()


# In[3]:


### Read the file required
trans = pd.read_csv('C:\\Users\\hp\\Desktop\\Madhav\\Tide_test_DS\\Tide_test_DS\\data_interview_test_v2.csv',delimiter=':')


# In[4]:


trans


# In[5]:


### Print the statistics about the data read
print ("The shape of data is as follows :", trans.shape)


# ### 1. Looking for Missing values

# In[6]:


print ("Give the information on the data required :",trans.info())


# #### No Missing values found : Missing value Treatment not required
# 
# ### 2. Looking for Outliers

# In[7]:


print ("Describe the data :", trans.describe())


# In[8]:


boxplot = trans.boxplot(column=['DateMappingMatch', 'AmountMappingMatch', 'DescriptionMatch',
       'DifferentPredictedTime', 'TimeMappingMatch', 'PredictedNameMatch',
       'ShortNameMatch', 'DifferentPredictedDate', 'PredictedAmountMatch',
       'PredictedTimeCloseMatch'], rot=90)


# #### No Outliers found : Outlier Treatment not required

# ### 3. Create target variable for the model
# 
# #### Target definition: If the  matched transaction id is equal to feature transaction id then 1 else 0

# In[9]:


trans['target'] = trans.apply(lambda x: 1 if (x.matched_transaction_id==x.feature_transaction_id)  else 0,axis=1)
target = trans["target"].value_counts()
target


# In[10]:


boxplot = trans[trans['target'] == 0].boxplot(column=['DateMappingMatch', 'AmountMappingMatch', 'DescriptionMatch',
       'DifferentPredictedTime', 'TimeMappingMatch', 'PredictedNameMatch',
       'ShortNameMatch', 'DifferentPredictedDate', 'PredictedAmountMatch',
       'PredictedTimeCloseMatch'], rot=90)


# In[11]:


boxplot = trans[trans['target'] == 1].boxplot(column=['DateMappingMatch', 'AmountMappingMatch', 'DescriptionMatch',
       'DifferentPredictedTime', 'TimeMappingMatch', 'PredictedNameMatch',
       'ShortNameMatch', 'DifferentPredictedDate', 'PredictedAmountMatch',
       'PredictedTimeCloseMatch'], rot=90)


# #### If we compare the box plot for data where target = 0 and where target = 1 , we find that there is a huge difference in values of the feature DifferentPredictDate. 
# 
# #### If  DifferentPredictDate is 0 then it is most likely the target is 0 ( in correct match) . This might come out as an important variable for model

# In[12]:


####  delete unnecessary columns

trans.drop(['receipt_id','company_id','matched_transaction_id','feature_transaction_id'],inplace=True,axis=1)
trans.head()


# ### 4. One hot encoding
# 
# 

# ### No need to convert the categorical variable to one hot dummy encoding as it only has 1,0 as values

# ### Validation base: Keeping 20% of the data aside for validation

# In[13]:


from sklearn.model_selection import train_test_split

model_data, val_data = train_test_split(trans, test_size=0.2)


# In[14]:


model_data.shape


# In[15]:


val_data.shape


# In[16]:


model_data["target"].value_counts()


# In[17]:


Y_val = val_data['target']
val_data.drop(['target'],inplace=True,axis=1)
val_data.head()


# ### 5. Balance Imbalance Class
# 
# ### This is a case of imbalance classification problem. 
# 
# #### The target here is 0 and 1 hence it is a classification problem
# 
# #### An imbalanced classification problem is an example of a classification problem where the distribution of target is biased or skewed. 
# #### Here the target has 7.1% amd 82.9% distribution
# 
# #### There are various methods to tackle this . I chose Oversampling than undersampling as the number of data points available are less
# 
# #### Random oversampling involves randomly duplicating examples from the minority class and adding them to the training dataset.

# In[18]:


model_majority = model_data[model_data['target']==0.0]
print(model_majority.shape)

model_minority = model_data[model_data['target']!=0.0]
print(model_minority.shape)


# In[20]:



from sklearn.utils import resample
# Oversample minority class
model_minority_oversampled = resample(model_minority, 
                                 replace=True,   
                                 n_samples=693*5,     # to get 30% minority class
                                 random_state=123) # reproducible results
 
# # Combine majority class with oversampled minority class
model_oversampled = pd.concat([model_minority_oversampled, model_majority])
 
model_oversampled.target.value_counts()


# In[21]:


target = model_oversampled['target']
model_oversampled.drop(['target'],inplace=True,axis=1)
model_oversampled.head()


# ### Test Train Split
# 
# ####  70:30 split is preferable in most cases, but due to less datapointd in  training set I am are using 80:20

# In[22]:


from sklearn.model_selection import train_test_split
# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(model_oversampled, 
                                                    target, 
                                                    test_size = 0.2, 
                                                    random_state = 0,stratify = target)

# Show the results of the split
print ("Training set has  samples", X_train.shape[0])
print ("Testing set has  samples  ",X_test.shape[0])


# ### 6. Trying out a Linear model in Logistic Regression

# In[23]:



from sklearn import linear_model

clf_A =  linear_model.LogisticRegressionCV(solver='lbfgs',random_state=40)
learner_A = clf_A.fit(X_train, y_train)


# #### Making the prediction on test data

# In[24]:



ypred = learner_A.predict(X_test,)

print(ypred)

#### Round off values i.e. 1 for values greater than 0.5 probability
predictions = [round(value) for value in ypred]

#### precison & Recall

from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))


# #### Make the prediction on validation data

# In[25]:



ypred_val = learner_A.predict(val_data,)

print(ypred_val)

#### Round off values i.e. 1 for values greater than 0.5 probability
predictions_val = [round(value) for value in ypred_val]

#### precison & Recall

from sklearn.metrics import classification_report

print(classification_report(Y_val, predictions_val))


# ### 7. Trying out a Decision Tree model in Random Forest

# In[26]:



from sklearn.ensemble import RandomForestClassifier

clf_B =  RandomForestClassifier(random_state=40)
learner_B = clf_B.fit(X_train, y_train)


# #### Making the prediction on test data

# In[27]:



ypred_rf = learner_B.predict(X_test,)

print(ypred_rf)

#### Round off values i.e. 1 for values greater than 0.5 probability
predictions_ref = [round(value) for value in ypred_rf]

#### precison & Recall

from sklearn.metrics import classification_report

print(classification_report(y_test, predictions_ref))


# #### Make the prediction on validation data

# In[28]:



ypred_rf_val = learner_B.predict(val_data,)

print(ypred_rf_val)

#### Round off values i.e. 1 for values greater than 0.5 probability
predictions_rf_val = [round(value) for value in ypred_rf_val]

#### precison & Recall

from sklearn.metrics import classification_report

print(classification_report(Y_val, predictions_rf_val))


# #### We are doing better than Logistic model , lets try a boosting model

# ### 8. Trying out a Boosted Decision Tree model in XG Boost

# In[29]:


import xgboost as xgb
from xgboost import XGBClassifier
from xgboost.sklearn import XGBClassifier

clf_c = XGBClassifier(eval_metric='logloss',random_state = 40)
learner_C=clf_c.fit(X_train, y_train)


# In[30]:


# import xgboost as xgb
# from xgboost import XGBClassifier
# from xgboost.sklearn import XGBClassifier

# clf_c = XGBClassifier(base_score = 0.5,booster = 'gbtree',colsample_bylevel=1,colsample_bytree=0.5,gamma=0,
#                       learning_rate=0.05,max_delta_step = 0 ,max_depth = 5, min_child_weight=5,missing = None,n_estimators = 200,
#                       n_jobs = -1,nthread = None,objective = 'binary:logistic',eval_metric='logloss',
#                       random_state = 0,reg_alpha=1,reg_lambda = 1,scale_pos_weight =1,seed=None,subsample = 1)
# learner_C=clf_c.fit(X_train, y_train)


# #### Making the prediction on test data

# In[31]:



ypred_xgb = learner_C.predict(X_test,)

print(ypred_xgb)

#### Round off values i.e. 1 for values greater than 0.5 probability
predictions_xgb = [round(value) for value in ypred_xgb]

#### precison & Recall

print(classification_report(y_test, predictions_xgb))


# #### Make the prediction on validation data

# In[32]:



ypred_xgb_val = learner_C.predict(val_data,)

print(ypred_xgb_val)

#### Round off values i.e. 1 for values greater than 0.5 probability
predictions_xgb_val = [round(value) for value in ypred_xgb_val]

#### precison & Recall

print(classification_report(Y_val, predictions_xgb_val))


# #### We are doing better than both Logistic and Random forest model, lets fine tune
# 
# ### 9. Parameter tuning
# 
# 

# In[33]:


params = {'n_estimators' : [100,150,200,250,300], # Number of trees in xg boost
         'min_child_weight' : [1,3,5,7],
         'reg_alpha' : [0,0.05,0.1,0.5,1],
         'subsample' : [0.9,0.8,0.7],
         'colsample_bytree' : [0.6,0.7,0.8],
         'max_depth' : [3,4,5,6], # Maximum number of levels in tree
         'learning_rate' : [0.05,0.1,0.5,1]} 

ind_params = {'base_score' : 0.5,'booster' : 'gbtree','colsample_bylevel': 1,'gamma' : 0,
                     'max_delta_step' : 0 ,'n_jobs' : -1,'objective' : 'binary:logistic','eval_metric' : 'logloss',
                      'random_state' : 40,'reg_lambda' : 1,'scale_pos_weight' : 1,'seed' : 9873}


# #### Params are the featured which are being tuned.
# #### ind_params are the features which aren't being tuned

# In[34]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
from sklearn.model_selection import RandomizedSearchCV
xgb =  XGBClassifier(**ind_params)
# Random search of parameters, using 2 fold cross validation, 
# search across 150 different combinations
xgb_random = RandomizedSearchCV(estimator = xgb, param_distributions = params, n_iter = 150, cv = 2, verbose=2)
# Fit the random search model
xgb_random.fit(X_train, y_train)
print (xgb_random.best_params_)


# #### Making the prediction on test data

# In[35]:


# Make predictions using the unoptimized and model
best_random = xgb_random.best_estimator_

# predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_random.predict(X_test)

print(classification_report(y_test, best_predictions))


# #### Make the prediction on validation data

# In[36]:



ypred_best_val = best_random.predict(val_data,)

print(ypred_best_val)

#### Round off values i.e. 1 for values greater than 0.5 probability
predictions_best_val = [round(value) for value in ypred_best_val]

#### precison & Recall

print(classification_report(Y_val, predictions_best_val))


#  ### 10.Variable importance

# In[37]:


best_random.feature_importances_


# In[38]:


var_imp = pd.DataFrame({'importance_in_model' : best_random.feature_importances_,'variable_name' : X_train.columns})
var_imp = var_imp.sort_values(['importance_in_model'],ascending = False).reset_index(drop = True)
print(var_imp)


# In[39]:


# Plot the feature importances of the forest
plt.figure(figsize=(12,8))
plt.title("Feature importances")
plt.xticks(rotation='vertical')
plt.bar(var_imp['variable_name'].values, var_imp['importance_in_model'].values, align='center', alpha=0.5)


# #### As predicted before DifferentPredictedTime came out to be the best feature

# ### Next Steps on improving the model
# 
# 1. Require more data points for better training of the model
# 2. Try More Sampling Technique like SMOTE
# 3. Try Random over sampling by playing around with minority percentage between (20%-35%)
# 4. Try More advanced Boosting techniques like XGBoost with class weights 
# 5. Improve by tuning more parameters on different set of values

# ### Deployment Tips:
# 1. Save the best model (best_random) as pickle file
# 2. Automate the  process and just predict
# 

# In[ ]:




