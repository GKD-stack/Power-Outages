#!/usr/bin/env python
# coding: utf-8

# # Power Outages Causation Classification

# # Summary of Findings
# 
# 
# ### Introduction
# The prediction problem I am attempting is "predict the cause of a major power outage." This is a classification problem. I am attempting to understand what was the main cause of a particular power outage using the information provided so the target variable is "CAUSE.CATEGORY". I evaluate the following models with the F1 score because because it achieves a balance between percision and recall while accounting for the imbalance in cause category (which accuracy doesn't). 
# 
# ### Baseline Model
# **Model:** Decision Tree Classifier.  
# **7 Features:**
# 
#     - Year (Ordinal)
#     - Month (Ordinal)
#     - US State (Nominal)
#     - Climate Region (Nominal)
#     - Population (Quantitative)
#     - Percent Water Inland (Quantitative)
#     - Area Percent Urban (Quantitative) 
# **Performance:**
# I used the F1 score as the primary metric for classification. I didn't deem percision or recall as being more relevant in this particular context so I picked the F1 score as a harmonic mean. I didn't choose accuracy because of the imbalances in the causes of power outages. 
# 
# I also broke down F1-score by weighted, micro, and macro to provide additional insight. For this particular case, I believe that the macro F1 score is most relevant because it weighs each class equally, directly addressing the class imbalance whereas micro weighs each observation equally, which poses the same problem as using accuracy. 
# 
#     - F1 score (macro): 0.35309892523229475
#     - F1 score (micro): 0.5859375
#     - F1 score (weighted): 0.5788115441845282
# 
# Since the higher the F1 score is the better, my model is currently ok according to (https://stephenallwright.com/good-f1-score/). However, it must be compared to other models trained and evaluated on the same tests for a more comprehensive evaluation. 
# 
# ### Final Model
# **Features:**
# These are the features I engineered in my model:  
# - **Engineered:**
#     - Month: Based off of project 3, it was clear there were more outages during summer monthes. Therefore, I used a function transformer to transform this variable into a 1 if it was between 5 an 8 months and 0 otherwise. 
#     - Urban: Based off of project 3, there was a positive correlation between larger states on the coast and the number of outages. Therefore, if a state had an urban area percentage that was above average in this dataset, I used a Binarizer to give it a 1, otherwise it was a 0.  
# - **One Hot Encoded:**
#     - Year: I included this a categorical variable because during certain years (based off of project 3), there were a series of events (ie natural disasters) that caused significantly more outages during certain years than others. 
#     - US State: I included this because certain states experienced power outages significantly more than others due to factors such as their size, customers, and geographic location. 
#     - NERC Region: I included this because the NERC region depends on the geographic area of a state which is highly correlated to the number and type of outage that can occur within an area. 
#     - Climate Region: I included this because the climate of a certain location has a correlation to the type of outage that will occur. Recent emperical examples of Texas and Florida further support this. 
# - **Passed Through:**
#     - Total Sales: I included this because this variable reveals the total electrical consumption in a particular state. If this consumption exceeds limits, it could cause a particular type of outage. 
#     - Total Customers: I included this because as this number rises, it imposes additional pressure on an electrical grid which could cause a particular type of outage. 
#     - Population: I included this because based off of project 3, it was clear that larger (more population) states had more outages likely due to more demand which could cause a particular type of outage. 
#     - Urban Cluster Area: I included this for the same reason I engineered the Urban variable as described above. 
#     - Percent Land: I included this for a similar reason as population. Although its clear that larger states such as California have more outages, the reason why (population, size) isn't clear. But percent land is a viable proxy for a "large" state because its still correlated with a larger population and more expansive electrical grid. 
#     - Percent Water Total: I included this because the amount of water area in a state is correlated to specific types of causes that depend on natural disasters. 
#     - Percent Water Inland: I included this as a balancing variable against percent land because it is plausable that a state may have a lot of land but much of it may be unlivable because it is occupied by bodies of water. For example, even if a state is large, it shouldn't be more likely to have a particular type of outage caused by demand because it also has a high percent of water inland. 
#     
# **Model Selection**
# 
# - **Decision Tree**
#     - **Best Parameters**
#         - {'tree__criterion': 'gini','tree__max_depth': 18,'tree__min_samples_split': 3}
#     - **Evaluation Metrics**
#         - F1 score (macro): 0.37658562670222623
#         - F1 score (micro): 0.6558441558441559
#         - F1 score (weighted): 0.6426739247522872
# 
# - **Random Forest**
#     - **Best Parameters**
#         - {'rf__criterion': 'gini', 'rf__max_depth': 10, 'rf__min_samples_split': 2, 'rf__n_estimators': 120}
#     - **Evaluation Metrics** 
#         - F1 score (macro): 0.3087639653562575
#         - F1 score (micro): 0.7142857142857143
#         - F1 score (weighted): 0.65112615579147020.6511261557914702
#         
# - **KNN**
#     - **Best Parameters**
#         - {'knn__algorithm': 'ball_tree', 'knn__n_neighbors': 19,     'knn__weights': 'distance'}
#     - **Evaluation Metrics**
#         - F1 score (macro): 0.39082644712896814
#         - F1 score (micro): 0.6948051948051948
#         - F1 score (weighted): 0.6637773715122837
# 
# <center><img src="p5.png" width="80%"></center>
# 
# **Chosen Model**
# 
# All the models had similar F1 scores. I choose a random forest classification model because I believe this is the most accurate since it has utilized several decision trees, which would mitigate bias and error that could arise during the training process. Furthermore, alongside its accuracy, the random forest can still capture non linear relationships, which is ideal for phenomana such as weather. 
# 
# 
# ### Fairness Evaluation
# **Parity Measure:** False Negative Rate. I chose this to understand whether the predicted cause category for a state is the same for Democratic states and Republican states. 
# 
# **Null Hypothesis:** My model is fair. The false negative rate for states with a Republican control as the same as states with a Democratic control. 
# 
# **Alternative Hypothesis:** My model is unfair. The false negative rate for Republican controlled states is higher.
# 
# **Justification of Measure and Hypothesis**: I believe that states with a Republican control are more likely to have a false negative rate  because they are less likely oriented towards data collection efforts due to their (and their voting bases') lack of emphasis on using scientific techniques as preventative measures. So Democrat controlled states are also more likely to provide data and hence I believe a higher proportion of particular causes will be correctly classified so they will have a lower false negative rate.
# 
# I did not select demographic parity because we cannot assume that the proportion of times the classifier predicts the cause correctly is independent of political control because Democratic states geographic locations are correlated with cause as learned through project three.
# 
# **Political Affiliation Data**: I found a dataset from https://worldpopulationreview.com/state-rankings/states-by-political-party that highlights each states political affiliation through data that includes "legislative majority paired with governor control (as seen in the table above), party affiliation of each state's governor, senate, and house, and 
# percentage of adults who identify as a democrat, republican, or neither".
# 
# **Results**: With a p value of .147 through 1000 iterations in a permutation test, I fail to reject the null hypothesis that my model is fair for Republican states and Democratic states using a parity measure of false negative rate. 

# # Code

# In[689]:


import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'  # Higher resolution figures")


# ### Baseline Model

# Load the data

# In[2]:


df = pd.read_excel("outage.xlsx")


# Clean the dataset to make it usable

# In[3]:


#remove the first 4 rows
df.drop(index=df.index[:4], inplace=True)
#make the variable names row the names of the columns in the df
df = df.rename(columns=df.iloc[0]).loc[1:]
#drop row with var names and extra row with no information
df.drop(index=df.index[:2], inplace=True)
#reset the index
df.reset_index(inplace=True)
df = df.drop(['index', 'variables','OBS'], axis=1)
df.head()


# Decide what variables to keep

# In[4]:


df.columns


# I will use year and month since some of the causes are related to time. I'll also use US state, NERC region, climate region, total sales, total customers, population, urban area percentage, urban cluster percentage, percent land, percent water total, and percent water inland. All of these variables will be available at the time before an outage actually occurs. 

# Limit dataframe to relevant columns

# In[248]:


df = df[['YEAR', 'MONTH', 'U.S._STATE', 'NERC.REGION',
       'CLIMATE.REGION', 'TOTAL.SALES',
       'TOTAL.CUSTOMERS', 'POPULATION', 
       'AREAPCT_URBAN', 'AREAPCT_UC', 'PCT_LAND', 'PCT_WATER_TOT',
       'PCT_WATER_INLAND','CAUSE.CATEGORY']]


# In[6]:


df.head()


# **Feature Transformation**
# 
# Now I will transform each of these variables for the random forest classifier. 
# 
# I will do one hot encoding for year, month, state, NERC region, and climate region. 
# 
# Before that, I'll make scatterplots for each quantitative variable to see if the underlying relationship is linear or if the column needs to be transformed. -----not done

# In[7]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score


# In[698]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[690]:


from sklearn import metrics


# Fill in missing values

# In[434]:


quant_col = ["TOTAL.SALES", "TOTAL.CUSTOMERS", "POPULATION", "AREAPCT_URBAN", "AREAPCT_UC", "PCT_LAND", "PCT_WATER_TOT", "PCT_WATER_INLAND"]
for c in quant_col:
    df[c] = df[c].fillna(df[c].mean())
cat_col = ['YEAR', 'MONTH', 'U.S._STATE', 'NERC.REGION', 'CLIMATE.REGION']
for c in cat_col: 
    num_null = df[c].isna().sum()
    fill_values = df[c].dropna().sample(num_null, replace=True)
    fill_values.index = df.loc[df[c].isna()].index
    df = df.fillna({c: fill_values.to_dict()}) 


# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

# In[764]:


X = df[['YEAR', 'U.S._STATE', 
       'CLIMATE.REGION', 'MONTH', 'POPULATION','PCT_WATER_INLAND', "AREAPCT_URBAN"]]
y = df['CAUSE.CATEGORY']
X_train, X_test, y_train, y_test = train_test_split(X, y)
preproc = ColumnTransformer(
    transformers = [
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['YEAR', 'U.S._STATE', 
       'CLIMATE.REGION', 'MONTH']),
        ('stan quant', StandardScaler(), ['POPULATION','PCT_WATER_INLAND', "AREAPCT_URBAN"])

    ]
)
pl = Pipeline([
    ('preprocessor', preproc), 
    ('tree', DecisionTreeClassifier())
])
pl.fit(X_train, y_train)
base_y_pred = pl.predict(X_test)
base_f1_macro = f1_score(y_test,  base_y_pred, average='macro')
base_f1_micro = f1_score(y_test,  base_y_pred, average='micro')
base_f1_w = f1_score(y_test,  base_y_pred, average='weighted')
print(base_f1_macro)
print(base_f1_micro)
print(base_f1_w)
metrics.plot_confusion_matrix(pl, X_test, y_test);


# In[703]:


# cm = metrics.plot_confusion_matrix(pl, X_test, y_test);
# ConfusionMatrixDisplay(confusion_matrix=cm,
#                                         xticks_rotation={"vertical"})


# ### Final Model

# Creating Features

# In[704]:


from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import FunctionTransformer


# In[784]:


preproc = ColumnTransformer(
    transformers = [
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['YEAR', 'U.S._STATE', 'NERC.REGION', 'CLIMATE.REGION'])
        ,('summer', FunctionTransformer(func = lambda d: d.transform(lambda x: x.apply(lambda n: 1 if (n>5) and (n<8) else 0))) , ['MONTH'])
        , ('urban', Binarizer(threshold=np.percentile(df["AREAPCT_URBAN"], 0.5)), ["AREAPCT_URBAN"]),
    ],remainder='passthrough'
)


# In[796]:


X_train.columns


# In[785]:


X = df.drop("CAUSE.CATEGORY", axis=1)
# X = df[['U.S._STATE', 'CLIMATE.REGION', "MONTH", "AREAPCT_URBAN", "POPULATION"]]
y = df['CAUSE.CATEGORY']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


# In[707]:


from sklearn import metrics


# Choosing a Model

# ### Decision Tree

# In[527]:


#decision tree
from sklearn.tree import DecisionTreeClassifier
import itertools
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV


# **Note**: I tried to run through all of these hyperparameters, but they took over 7 hours and still were not finished so eventually I had to move on with the 3 that are uncommented. 

# In[786]:


pl = Pipeline([
    ('preprocessor', preproc), 
    ('tree', DecisionTreeClassifier())
])

pl.fit(X_train, y_train)


hyperparameters = {
    'tree__max_depth': [10, 13, 14, 16, 18, 20], 
    'tree__min_samples_split': [2,3, 5, 7, 8, 10],
    'tree__criterion': ['gini', 'entropy']
#     'tree__min_samples_leaf':[5,7,10,20], 
#     'tree__max_features':[2,4,6,8,10,12,14, None], 
#     'tree__min_weight_fraction_leaf':[0.1,0.2,0.4, None], 
#     'tree__class_weight':[.05, .07, .1, 0.2, 0.3, None]
}

grids = GridSearchCV(pl, param_grid=hyperparameters, cv=5, scoring='f1_macro',error_score='raise')

grids.fit(X_train, y_train)

tree_y_pred = grids.predict(X_test)

tree_f1_macro = f1_score(y_test,  tree_y_pred, average='macro')

tree_f1_micro = f1_score(y_test,  tree_y_pred, average='micro')

tree_f1_w = f1_score(y_test,  tree_y_pred, average='weighted')

print(tree_f1_macro)
print(tree_f1_micro)
print(tree_f1_w)


# In[787]:


metrics.plot_confusion_matrix(pl, X_test, y_test);


# In[788]:


grids.best_params_


# ### Random Forest

# In[589]:


from sklearn.ensemble import RandomForestClassifier


# In[653]:


RandomForestClassifier().get_params().keys()


# In[789]:


pl = Pipeline([
    ('preprocessor', preproc), 
    ('rf', RandomForestClassifier())
])


hyperparameters = {
    'rf__n_estimators':[80,100,120],
    'rf__max_depth': [10], 
    'rf__min_samples_split': [2],
    'rf__criterion': ['gini']
}


grids = GridSearchCV(pl, param_grid=hyperparameters, cv=5, scoring='f1_micro',error_score='raise')

grids.fit(X_train, y_train)

print(grids.best_params_)

rf_y_pred = grids.predict(X_test)

rf_f1_macro = f1_score(y_test,  rf_y_pred, average='macro')

rf_f1_micro = f1_score(y_test,  rf_y_pred, average='micro')

rf_f1_w = f1_score(y_test,  rf_y_pred, average='weighted')

print(rf_f1_macro)
print(rf_f1_micro)
print(rf_f1_w)


# ### K Nearest Neighbors

# In[603]:


from sklearn.neighbors import KNeighborsClassifier


# In[648]:


KNeighborsClassifier().get_params().keys()


# In[790]:


pl = Pipeline([
    ('preprocessor', preproc), 
    ('knn', KNeighborsClassifier())
])

pl.fit(X_train, y_train)


hyperparameters = {
    'knn__n_neighbors': np.arange(1,20), 
    'knn__weights': ['uniform','distance'],
    'knn__algorithm': ['ball_tree','kd_tree','brute','auto']
}

grids = GridSearchCV(pl, param_grid=hyperparameters, cv=5, error_score='raise')

grids.fit(X_train, y_train)

print(grids.best_params_)

knn_y_pred = grids.predict(X_test)

knn_f1_macro = f1_score(y_test,  knn_y_pred, average='macro')

knn_f1_micro = f1_score(y_test,  knn_y_pred, average='micro')

knn_f1_w = f1_score(y_test,  knn_y_pred, average='weighted')

print(knn_f1_macro)
print(knn_f1_micro)
print(knn_f1_w)


# ### Comparison of All Models

# In[793]:


viz = [['knn',knn_f1_macro,'macro'], \
       ['knn',knn_f1_micro,'micro'], \
       ['knn',knn_f1_w,'weighted'], \
       
       ['random forest',rf_f1_macro,'macro'], \
       ['random forest',rf_f1_micro,'micro'], \
       ['random forest',rf_f1_w,'weighted'], \
       
       ['decision tree',tree_f1_macro,'macro'], \
       ['decision tree',tree_f1_micro,'micro'], \
       ['decision tree',tree_f1_w,'weighted']]
       
viz = pd.DataFrame(viz)
viz.columns = ["Model", "Score","F1 Type"]
viz


# In[794]:


sns.barplot(x='Model',y='Score', hue="F1 Type", data=viz)


# All of the models produced rather similar output. I would go with Random Forest because it has the highest micro score and has reduced the amount of variation and bias that could originate from random trials by being trained on many trees.

# ### Fairness Evaluation

# **Parity Measure:** False Negative Rate. I chose this to understand whether the predicted cause category for a state is the same for states with Democratic control in the House as well as states with Republican control in the house. 
# 
# **Null Hypothesis:** My model is fair. The false negative rate for states with a Republican control as the same as states with a Democratic control. 
# 
# **Alternative Hypothesis:** My model is unfair. The false negative rate for Republican controlled states is higher.
# 
# **Justification of Measure and Hypothesis**: I believe that states with a Republican control in the House are more likely to have a false negative rate  because they are less likely oriented towards data collection efforts due to their (and their voting bases') lack of emphasis on using scientific techniques as preventative measures. So Democrat controlled states are also more likely to provide data and hence I believe a higher proportion of particular causes will be correctly classified so they will have a lower false negative rate.
# 
# I did not select demographic parity because we cannot assume that the proportion of times the classifier predicts the cause correctly is independent of political control because Democratic states geographic locations are correlated with cause as learned through project three.

# In[725]:


poli_df = pd.read_csv('poli.csv')
poli_df.head()


# In[726]:


df.head()


# In[727]:


#left merge regular df with this one 
fairness_df = df.merge(poli_df, how='left', left_on='U.S._STATE', right_on="state")
fairness_df.head()


# In[734]:


#get predicted y values for whole dataframe
pl = Pipeline([
    ('preprocessor', preproc), 
    ('rf', RandomForestClassifier())
])
hyperparameters = {
    'rf__n_estimators':[80],
    'rf__max_depth': [10], 
    'rf__min_samples_split': [2],
    'rf__criterion': ['gini']
}
grids = GridSearchCV(pl, param_grid=hyperparameters, cv=5, scoring='f1_micro',error_score='raise')
grids.fit(X_train, y_train)
whole_pred = grids.predict(fairness_df[list(X_test.columns)])
whole_pred


# In[736]:


fairness_df['predicted cause'] = whole_pred
fairness_df


# In[743]:


(
    fairness_df
    .groupby('control')
    .apply(lambda x: 1 - metrics.recall_score(x['CAUSE.CATEGORY'], x['predicted cause'], average="macro"))
    .plot(kind='bar', title='False Negative Rate by Political Control')
);


# This initial bar graph confirms my hypothesis that states with Republican control in the House have a higher false negative rate. 

# In[748]:


#keep on republicans and demograts in data
rd_df = fairness_df[(fairness_df['control']=="Democrat")|(fairness_df['control']=="Republican")]


# In[751]:


obs_ser = rd_df.groupby('control').apply(lambda x: 1 - metrics.recall_score(x['CAUSE.CATEGORY'], x['predicted cause'], average="macro"))
print(obs_ser)
obs = obs_ser.diff().iloc[-1]
obs


# In[759]:


diffs = []
for i in range(1000):
    s = (
        rd_df
        .assign(control=rd_df.control.sample(frac=1.0, replace=False).reset_index(drop=True))
        .groupby('control')
        .apply(lambda x:1 - metrics.recall_score(x['CAUSE.CATEGORY'], x['predicted cause'], average="macro"))
        .diff()
        .iloc[-1]
    )
    
    diffs.append(s)


# In[761]:


plt.figure(figsize=(10, 5))
pd.Series(diffs).plot(kind='hist', ec='w', density=True, bins=15, title='Difference in False Negative Rate (Democrat - Republican)')
plt.axvline(x=obs, color='red', label='observed false negative rate')
plt.legend(loc='upper left');


# In[767]:


(diffs >= obs).sum() / 1000 #p value is not significant 

