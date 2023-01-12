#!/usr/bin/env python
# coding: utf-8

# # Where and when do major power outages tend to occur?
# The data is downloadable [here](https://engineering.purdue.edu/LASCI/research-data/outages/outagerisks).
# 
# A data dictionary is available at this [article](https://www.sciencedirect.com/science/article/pii/S2352340918307182) under *Table 1. Variable descriptions*.

# # Summary of Findings
# 
# ### Introduction
# For this project, I'll be exploring where and when do major power outages tend to occur. I will use the dataset provided by Purdue Engineering, which provide a variety of columns describing time, location, attributes about the location. The overall question has been broken down into these subcomponents highlighted further in hypothesis tests for clarity. 
# 
# ### Cleaning and EDA
# Using the questions above, the dataset was filtered down to relevant columns and exploratory data analysis through various charts and sample statistics was done for each subquestion.
# 
# Specifically, to understand if there an association between location and outages, the outages, their type and total amount were aggregrated per state. Maps were created to break down the number of outages per state and highlight which causes were more prevalent in particular state. Then to understand is there an association between time and causes, I created separate dataframes grouped by year and month to understand during which years and during which months outages were more common. Additionally, I broke down the causes per outages by year also to understand one of the trends illustrated by the year and number of outages line plot. 
# 
# ### Assessment of Missingness
# Each column that I utilized during a hypothesis test was inspected manually to account for missing values that were not in np.NaN format and explanations are provided below. Imputations were performed according to criteria and strategies described in lecture. 
# 
# Technically only 1 column, month, that I used in the hypothesis tests had missing values. However, I did analyze the missingness of a variety of other columns that could be relevant to my overall question. For month, I initially inferred that its missingness was missing completely at random. However, based on the trends in the EDA portion, I performed a permutation test to understand whether the missingness of month could be attributed to the column 'year'. 
# 
# Here is the core information about this permutation test:
# * Permutation Test to Assess Missingness of Month: 
#     * Null Hypothesis: Missing 'MONTH' values are missing completely at random.
#     * Alternative Hypothesis: Missing 'MONTH' values are dependent on another column.
#     * Test Statistic: Absolute difference in means
#     * Significance Level: .01
#     * Conclusion: The p value was 0 and we reject the null hypothesis in favor of the alternative that the missing month values are dependent on the year column. 
# 
# Another permutation test was performed to assess if the missingness of the 'customers affected' column was dependent 'outage duration'. Here is the core information about this permutation test: 
# * Permutation Test to Assess Missingness of Customers Affected:
#     * Null Hypothesis: The missingness of 'CUSTOMERS.AFFECTED' is not dependent on the 'OUTAGE.DURATION' column.
#     * Alternative Hypothesis: The missingness of 'CUSTOMERS.AFFECTED' is dependent on the 'OUTAGE.DURATION' column.
#     * Test Statistic: Kolmogorov-Smirnov
#     * Significance Level: .01
#     * Conclusion: The p value was 0 and we reject the null hypothesis in favor of the alternative that the missing customers affected values are dependent on the year outage duration. 
# 
# ### Hypothesis Test(s)
# Based on certain trends in the EDA, I ran the following hypothesis tests on the followng subquestions to answer the overarching question. In addition to understand when and where the major power outages occur, I delved into how causes varied by location and timing. 
# * Do states with a high number of total customers have more outages than states with a low number of total customers?
#     * Null Hypothesis: There is no relationship between a states total customers and the number of outages it has.
#     * Alternative Hypothesis: States with a higher number of total customers tend to experience a greater number of outages.
#     * Test Statistic: Signed mean difference between number of outages in states with a high total customers and low total customers.
#     * Significance Level: 0.01.
#     * Conclusion: Using a significance level of .01 and our p value of 0, we can reasonably reject the null hypothesis in favor of the alternative hypothesis that states with larger total customers tend to have more outages.
# 
# * Are outages more common in the summer months?
#     * Null Hypothesis: There is no relationship between an outage occuring and what month it is.
#     * Alternative Hypothesis: Outages are more likely to occur during summer (May to August) versus other seasons.
#     * Test Statistic: Signed mean difference between number of outages in summer and number of outages in other seasons.
#     * Significance Level: 0.01.
#     * Conclusion: Using a significance level of .01 and our p value of 0.0075, we can reasonably reject the null hypothesis in favor of the alternative hypothesis that outages are more likely to occur during summer monthes. This information can help governments and energy provides take additional preventative measures during summer.
#     
# * Are the causes of outages in summer different from the causes of outages in other seasons?
#     * Null Hypothesis: The causes of outages in summer are not different from the causes of outages in seasons that are not summer.
#     * Alternative Hypothesis: The causes of outages in summer are different from the causes of outages in seasons that are not summer.
#     * Test Statistic: TVD
#     * Significance Level: 0.01.
#     * Conclusion: Using a significance level of .01 and our p value of 0.0003, we can reasonably reject the null hypothesis in favor of the alternative hypothesis that the causes of outages in summer are different from the causes of outages in seasons that are not summer.
# 
# *Note: Small is defined is less than then the median number of customers per state and large is defined as the greater than or equal to the median number of total customers per state. A outage is classified as occuring in summer if it occured in months 5-8*
# 
# ### Conclusion
# During further analysis with this dataset, I would aim to understand how the commercial, residential, and industrial electrical consumption of a state influenced the number of outages it had and the type of outages it had. Furthermore, I would build upon the geographical attributes of a state such as general climate and inland water percentages influenced the number of and type of causes for outages that occured in that state. 
# 
# Some shortfalls in this current dataset are that I didn't fully explore what could have caused some of the spikes in outages and specific causes in certain states and time periods. There are likely confounding variables that have not been accounted for in this data analysis. 

# # Code

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'  # Higher resolution figures")


# In[48]:


pip install nbconvert


# In[45]:


pip install nbconvert[webpdf]


# In[2]:


pip install pandoc


# In[3]:


pip install openpyxl


# In[4]:


pip install chart_studio


# In[5]:


#libs for maps in EDA

import chart_studio.plotly as py
import plotly.offline as po
import plotly.graph_objs as pg
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy import stats

po.init_notebook_mode(connected = True)


# ### Cleaning and EDA

# #### The first primary step is loading the data.

# In[6]:


df = pd.read_excel("outage.xlsx")


# As we loaded in the data, it appears that the commentary on the excel file has made the first 4 rows primary NaN. The 5th row with index 4 has a row with the variable columns. In the next chunk of code, we'll remove the first 4 rows and set the 5th row as the column header.

# In[7]:


#remove the first 4 rows
df.drop(index=df.index[:4], inplace=True)
#make the variable names row the names of the columns in the df
df = df.rename(columns=df.iloc[0]).loc[1:]
#drop row with var names and extra row with no information
df.drop(index=df.index[:2], inplace=True)
#reset the index
df.reset_index(inplace=True)
#df


# In[8]:


df.head()


# To determine which columns to keep, we'll go into what the categories of causes are

# In[9]:


df['CAUSE.CATEGORY'].unique()


# Now we'll examine the columns to determine which to keep

# In[10]:


df.columns


# For each category we want to first see if some occured more in certain areas then others and the timings of the outage event. For this, the columns year, month, US state, nerc region, climate region, and outage duration. 
# 
# Then we see how the impact of each type of outage cause. We will use the columns outage duration, demand loss, and customers affected. 
# 
# To explore the relationship between electricity use and outages, the columns total price, sales, and customers per state will also be kept alongside commercial and industrial customers percent.

# In[11]:


df = df[["YEAR", "MONTH",'U.S._STATE',
       'CLIMATE.CATEGORY', 'CAUSE.CATEGORY','POSTAL.CODE', 
       'CAUSE.CATEGORY.DETAIL', 'OUTAGE.DURATION',
       'CUSTOMERS.AFFECTED', 'TOTAL.PRICE',
       'TOTAL.SALES', 'TOTAL.CUSTOMERS']]


# In[12]:


df.head()


# ### EDA on Cause Category
# 
# In the upcoming section, the cause category section will be analyzed to determine which causes are more prevalent compared to others. Further sections will explore the cause categories in relation to the subquestions highlighted above. 

# ### EDA: Is there an association between location and outages?
# 
# In this chunk of code, we grouped by US State, aggregated the causes of the outages into a list per state, add a column for number of outages per state and kept the postal code per state since it is a key paramter to making US maps later on.

# In[13]:


location_cause_df = df[['U.S._STATE','POSTAL.CODE','CAUSE.CATEGORY']]
location_cause_df

cause_categories = df['CAUSE.CATEGORY'].unique()
#get each state as the index and all the causes in a list 
location_cause_state = location_cause_df.groupby('U.S._STATE').agg(lambda x: list(x))
location_cause_state['POSTAL.CODE'] = location_cause_state['POSTAL.CODE'].apply(lambda lst: lst[0])
location_cause_state['Num Outages'] = location_cause_state['CAUSE.CATEGORY'].apply(lambda lst: len(lst))
location_cause_state.head()


# I used this link as a resource for making graphs
# 
# https://towardsdatascience.com/geographical-plotting-of-maps-with-plotly-4b5a5c95f02a

# Using plotly and the location state information provided in the original graph, this next chunk creates a map showing the number of outages in the US by state where the brighter a color is the more outages it has. You can also scroll over this graph to view the state name and the number of outages it had. 

# In[14]:


#this chunk will make a map showing the the relationship between the number of outages per state
data = dict(type = 'choropleth', 
     locations = list(location_cause_state['POSTAL.CODE']),  #locations will be index of df 
     locationmode = 'USA-states', 
     z = list(location_cause_state['Num Outages']), #this will vary for each cause 
     text = list(location_cause_state.index)) #index of df 

layout = dict(geo = {'scope':'usa'})

x = pg.Figure(data = [data], layout = layout)
x.update_layout(title_text=f"Number of occurances of outages in US by state")
po.iplot(x)


# This following chunk of code will produce multiple maps of outages by state per cause. The mechanism is to loop over each state, filter the causes to a specific cause, and create a map with the appropriate paramters

# In[15]:


#this chunk of code will loop through all the causes and product a map showing in which states it is most prevalent 
for cause in cause_categories: 
    #this function is for leaving only the specified cause in a list of various causes 
    def filter_cause(lst):
        return len([c for c in lst if c == cause])
    
    #this adds a column of number of cases by a particular cause per each state 
    sw_ser_df = location_cause_state.assign(num_cases = location_cause_state['CAUSE.CATEGORY'].apply(filter_cause))
    
    #this chunk creates a dictionary of data to be inputed to plotly to generate a US map
    data = dict(type = 'choropleth', 
         locations = list(sw_ser_df['POSTAL.CODE']),  #locations will be index of df 
         locationmode = 'USA-states', 
         z = list(sw_ser_df['num_cases']), #this will vary for each cause 
         text = list(sw_ser_df.index)) #index of df 
    
    #this line specifies the geo map's scope will be the US 
    layout = dict(geo = {'scope':'usa'})
    
    x = pg.Figure(data = [data], layout = layout)
    x.update_layout(title_text=f"Number of occurances of outages caused by {cause} in US by state")
    po.iplot(x)


# ### Map Analysis: Which states are outages more prevalent in (broken down by cause)?
# 
# * **Severe Weather**: Outages caused by severe weather were more common in Michigan, California, Texas. 
# * **Intentional Attack**: Outages caused by intentional attack were more common in Washington, Utah, and Delaware. 
# * **System Operability Disruption**: Outages caused by system operability disruption were more common in California, and Texas. 
# * **Equipment Failure**: Outages caused by equipment failure were most common in California-21. Other states had few to none. 
# * **Public Appeal**: Outages caused by public appeal were by far most common in Texas-17-, California-9-, and Arkansas-7-. 
# * **Fuel Supply Emergency**: Outages caused by fuel supply emergency were more common in California, New York, and Texas. 
# * **Islanding**: Outages caused by islanding were by far most common in California-28-, whereas other states had few to none. 

# ### EDA: Is there an association between time and causes?

# #### Does the number of outages reported increase over time?

# In the following chunk of code, a dataframe and histogram grouped by year and the number of outages is created to provide insight into how the number of outages reported has changed over the years 2000-2016.

# In[16]:


#get the total number of outages that occured per year -- histogram?
year_num_causes = df[["YEAR","MONTH","CAUSE.CATEGORY"]].groupby("YEAR").agg(lambda x:list(x))
#add column for num of outages 
year_num_outages = year_num_causes.assign(Num_Outages = year_num_causes['CAUSE.CATEGORY'].apply(lambda lst: len(lst)))
year_num_outages = year_num_outages.reset_index()
#display(year_num_outages.head())
year_num_outages.plot.line(x='YEAR', y='Num_Outages', title="Number of Outages Reported from 2000-2016")


# It appears that the number of outages steadily increased from 2000 to 2010 and had a sharp peak in 2012. Now we'll aim to understand where which months outages occur the most. 

# This following chunk of code groups the outages by month to undestand whether outages are more likely to occur during a particular season. Assume that 1=January and 12=December with other values corresponding similarily.

# In[17]:


month_outage = df[["MONTH", "CAUSE.CATEGORY"]].groupby("MONTH").agg(lambda x: len(list(x))).reset_index()
month_outage = month_outage.rename(columns={"CAUSE.CATEGORY":"Num_Outages"})
#display(month_outage)
month_outage.plot.line(x='MONTH', y='Num_Outages', title="Number of Outages Reported by Month")


# It appears that there are more outages occuring from months 5-8 during the summer season. A hypothesis test will be conducted later to affirm or deny this hypothesis.

# #### Does the kind of outage cause change over years and months?

# This following chunk of code creates dataframes grouped by month and year to eventually produce histograms that highlight which causes are more prevalent during particular years and months.

# In[18]:


#this needs a dataframe grouped by year and have different lines for num outages 
year_cause_df = df.copy(deep=True)
unique_causes = list(df["CAUSE.CATEGORY"].unique())
year_cause_df = year_cause_df.groupby("YEAR").agg(lambda x: list(x))[["CAUSE.CATEGORY"]]

#make a month cause df 
month_cause_df = df.copy(deep=True)
month_cause_df = month_cause_df.groupby("MONTH").agg(lambda x: list(x))[["CAUSE.CATEGORY"]]

#create a column for each cause with number of times it occured in a list 
year_cause_df.is_copy = True
for specific_cause in unique_causes:
    cause = specific_cause
    def count_spec_cause(lst):
        output = 0
        for c in lst:
            if c==cause:
                output+=1 
        return output 
    year_cause_df[cause] = year_cause_df['CAUSE.CATEGORY'].apply(count_spec_cause)
    month_cause_df[cause] = month_cause_df['CAUSE.CATEGORY'].apply(count_spec_cause)
    
# display(year_cause_df.head())
# display(month_cause_df.head())


# In[19]:


years = list(year_cause_df.index)
#go through each cause in the dataframe, get it as a list 
for cause in unique_causes:
    new_line_data = list(year_cause_df[cause])
    plt.plot(years, new_line_data, label=cause)
plt.title("Number of Outages by Cause from 2000-2016")
plt.legend()
plt.show()

months = list(month_cause_df.index)
for cause in unique_causes:
    new_line_data = list(month_cause_df[cause])
    plt.plot(months, new_line_data, label=cause)
plt.title("Number of Outages by Cause by Month")
plt.legend()
plt.show()


# For the line plot "Number of Outages by Cause from 2000-2016", it appears that severe weather caused significant outages in comparison to other causes consistently from 2003 to 2015. Intentional attacks also spiked from 2010-2014. This provides more insight into the line graph that showed the number of outages per year earlier. 
# 
# For the line plot "Number of Outages by Cause by Month", it appears that intentional attacks and severe weather generally occur more than other types. Severe weather attacks also tend to peak in the summer months from 5-8. 

# ### Assessment of Missingness

# Since we cannot assume that the each missing value is represented by a NaN, each columns unique values must be highlighted to discern if another particular value is used

# In[20]:


#For stylistic purposes, the print statements have been commented out
for col in df.columns:
    #print(col)
    unique_vals = df[col].unique()
    #print(unique_vals)


# #### The following variables had missing values upon manual inspection. 
# 

# MONTH CLIMATE.CATEGORY CAUSE.CATEGORY.DETAIL OUTAGE.DURATION 
# 
# CUSTOMERS.AFFECTED	TOTAL.PRICE	TOTAL.SALES

# #### Month: Used in Hypothesis Test
# Missing month values are **missing completely at random** because there is no reasonable relationship between any of the other columns and this one. We could check distribution of months to be sure, but the distribution will likely be skewed anyhow since it is more likely that the causes will occur in winter 
# 
# #### Climate Category: Not used
# Missing climate category values are **not missing at random** because it is likely that if one deems the climate of a region to be average or unnotable, they will likely not include it. Whereas, if the climate of a region is extreme somehow, it will likely be included. 
# 
# #### Cause Category Detail: Not used
# Missing cause category detail values are **missing at random** because how much the specific outage cause resembles the categorical cause will determine whether or not this field is populated. For example, if a specific outage matched a cause nearly directly, this description may not be filled since one could deem that the cause itself was sufficiantly descriptive about the outage.
# 
# #### Outage Duration: Not used
# Missing outage duration values are **not missing at random** because if outage values were extremely short, they were likely deemed not significant enough to be recorded. 
# 
# #### Customers Affected: Not used
# Missing customers affected values are either **not missing at random** or **missing at random**. They could be not missing at random because if the number of customers affected was marginal, the value could have been insiginificant and not have been recorded. On the other hand, the number of customers affected likely has an association with the type of cause (assuming that certain causes affect more people than others), therefore these value values could also be missing at random. 
# 
# #### Total Price and Total Sales: Not used
# These variables are being grouped together because they reveal similar information from similar sources. Missing values in both variables are **missing at random** because since both variables reveal information about the average monthly price and sales in a certain state, it is highly likely that the state and its corresponding attributes influence whether or not these variables were recorded. 

# ### Is the missingness of 'MONTH' dependent on any of the relevant columns?

# **Null Hypothesis**: Missing 'MONTH' values are missing completely at random. 
# 
# **Alternative Hypothesis**: Missing 'MONTH' values are dependent on another column. 
# 
# Test Statistic: Absolute difference in means
# 
# Significance Level: .01
# 
# To test whether the missingness of "MONTH" is dependent on any of the other relevant columns, we will perform a permutation test on "MONTH" and "YEAR" first because it is likely that the month in beginning years wasn't recorded for logistical reasons or prehaps there were years with significantly more outages where month wasn't recorded due to a lack of resources and emphasis on data collection

# This following chunk of code prepares the dataaframe for a permutation test, draws a density line plot for comparison of distrubtions of year when month is missing and when month isn't, and calculates the test statistic. 

# In[21]:


#prepare data frame
test_copy = df.copy(deep=True)
test_copy["Month Missing"] = test_copy["MONTH"].isna()
    
#draw histogram
(
    test_copy
    .groupby("Month Missing")["YEAR"]
    .plot(kind='kde', legend=True, title="Year by Missingness of Month")
);


# From previous plots, we know that there were the most amount of outages around 2012. This line plot reveals that month is missing for outages that occured primarily around 2000, the earlier outages in this dataset. Since the means of the two distributions are not overlapping, we will use the absolute difference in means as a test statistic for this graph.

# In[22]:


#calculate the observed test statistic
obs_ser = test_copy.groupby("Month Missing")["YEAR"].mean()
obs_test_stat = abs(obs_ser.iloc[0] - obs_ser.iloc[1])
obs_test_stat


# In[23]:


#do permutation to get null distributions
test_copy
null = []
for i in range(10000):
    perm_df = test_copy.copy(deep=True)
    #make a shuffled month missing column
    shuffled_month_missing = (
    perm_df['Month Missing']
    .sample(frac=1)
    .reset_index(drop=True) 
)
    #add shuffled column to dataframe
    perm_df['Month Missing'] = shuffled_month_missing
    
    #get null stat
    perm_ser = perm_df.groupby("Month Missing")["YEAR"].mean()
    perm_stat = abs(perm_ser.iloc[0] - perm_ser.iloc[1])
    
    #add null stat 
    null.append(perm_stat)


# In[24]:


#calculate the p value 
p_val = np.mean(null>=obs_test_stat)
p_val


# #### Conclusion
# Since the p value is below our significance level of .01, it is reasonable to reject the null hypothesis that the MONTH column is has values missing completely at random since the permutation tests highlight that the YEAR column distribution when MONTH is missing and when it isn't is significantly different. Therefore MONTH is missing at random and it is dependent on the YEAR column

# ### Is the missingness of 'CUSTOMERS.AFFECTED'  dependent on the 'OUTAGE.DURATION' column?
# 
# Since, none of the columns' missingness utilized in the hypothesis tests could be attributed to MAR, I have chosen a column with missing values 'customers affected' and will perform a permutation test to see whether it is dependent on 'outage duration', making it MAR. An intuitive explanation for why 'customers affected' could be dependent on 'outage duration' is that if the outage duration was miniscule then the number of customers affected was not collected since the impact on the customers was minimal. 
# 
# **Null Hypothesis**: The missingness of 'CUSTOMERS.AFFECTED' is not dependent on the 'OUTAGE.DURATION' column. 
# 
# **Alternative Hypothesis**: The missingness of 'CUSTOMERS.AFFECTED' is dependent on the 'OUTAGE.DURATION' column. 
# 
# Test Statistic: Kolmogorov-Smirnov
# 
# Significance Level: .01

# This following chunk of code produces a clear dataframe that highlights the columns we kept for this permutation and the columns created. 

# In[25]:


missingness_df = df[['CUSTOMERS.AFFECTED', 'OUTAGE.DURATION']]
missingness_df["Customers Affected Missing"] = missingness_df['CUSTOMERS.AFFECTED'].isna()
missingness_df.head()


# This following chunk of code plots how the outage duration distribution differs based on whether the customers affected values are missing or not.

# In[26]:


(
    missingness_df
    .groupby('Customers Affected Missing')['OUTAGE.DURATION']
    .plot(kind='kde', legend=True, title="Observed Outage Duration by Missingness of Customers Affected")
);


# Since the means of the two distributions are similar but the shape differs, we will use the Kolmogorov-Smirnov test statistic.

# In this next code chunk, we make series extracting the distributions plotted above, do permutation tests, and calculate the KS statistic and p value.

# In[27]:


#make outage durations for customers affected missing or not 
outage_cust_miss = missingness_df.loc[missingness_df['Customers Affected Missing'], 'OUTAGE.DURATION']

# 'father' when 'child' is not missing
outage_cust_not_miss = missingness_df.loc[~missingness_df['Customers Affected Missing'], 'OUTAGE.DURATION']

#permutation test and ks stat 
stats.ks_2samp(outage_cust_miss, outage_cust_not_miss)


# #### Conclusion
# This states that if the missingness of 'customers affected' is truly unrelated to the distribution of 'outage duration', then the chance of seeing two distributions that are as or more different than our two observed 'outage duration' distributions is 88%.
# 
# Additionally, if we set the significance value to .01, based on the p value, we reject the null in favor of the alternative hypothesis that there 'customers affected' is dependent on 'outage duration'. 

# ### Imputation for Missing Values
# I only imputed for missing values for columns that were utilized within the hypothesis tests. 

# #### Month: 
# Since it is likely that the month of certain incidents is missing at random and is dependent on the YEAR column, we will impute the missing month values with group means by that year.
# 
# So the if a outage has a missing month value and the year is 2010, we will fill that value with the mean of all the months in 2010. 

# In[28]:


#get the mean of the year that a month was found and a month was missing
month_year_df = df.copy(deep=True)
#make a df with the averages of each year 
year_month_avg_df = month_year_df.groupby("YEAR")["MONTH"].mean().reset_index()
display(year_month_avg_df)
#get all the years as a list to loop through
years_list = list(year_month_avg_df["YEAR"])


# In[29]:


#go through all years
#if the year is equal to that year and value is missing
for year in years_list:
    df.loc[(df["YEAR"]==year) & (df["MONTH"].isnull()),"MONTH"] = round(year_month_avg_df.loc[year-2000, "MONTH"])


# In[30]:


#check if there are any floats or missing values
df["MONTH"].unique()


# ### Hypothesis Test

# In this section, there are multiple hypothesis tests that answer the overall question of when and where do power outages tend to occur?
# 
# The hypothesis tests are:
# * Do states with higher total customers have more outages?
# * Are outages more common in the summer months?
# * Are the causes of outages in summer different from the causes of outages year around?

# ### HT1: Do states with higher total customers have more outages?
# 
# **Null Hypothesis**: There is no relationship between a states total customers and the number of outages it has. 
# 
# **Alternative Hypothesis**: States with a higher number of total customers tend to experience a greater number of outages. 
# 
# Test Statistic: Signed mean difference between number of outages in states with a high total customers and low total customers. 
# 
# Significance Level: 0.01. 
# 
# A column will be created to distinguish the whether states have a low or high number of total customers. The threshold will be the median number of total customers from all states. If a state's total customers is greater than its threshold, then it will be a deemed with a "high" customer population, otherwise it will have a "low" customer population. We will then run a permutation test with the above questions to answer this question. Specific steps are highlighted in the code itself.

# In[31]:


#make copy of dataset 
q1 = df.copy(deep = True)
#get the number of outages 
q1 = q1.groupby("U.S._STATE")[["CAUSE.CATEGORY"]].agg(lambda x: len(list(x)))
#update column name 
q1 = q1.rename(columns=
    {"CAUSE.CATEGORY":"Num Outages"})

# display(q1)
#get just total customers
total_customers = df[["TOTAL.CUSTOMERS", "U.S._STATE"]].groupby("U.S._STATE").mean()

# display(total_customers)
#merge the total customers into this from the original df 
merged_df =q1.merge(total_customers, how='inner'
              , left_index=True, right_index=True)

threshold = merged_df["TOTAL.CUSTOMERS"].median()

#use threshold to classify each population as large or small
merged_df["Customer Size"] = 'Small' 
merged_df.loc[merged_df["TOTAL.CUSTOMERS"]>=threshold, "Customer Size"] = 'Large'
merged_df

#let test statistic be the difference in mean of num outages between states 
# with large and small customer sizes 
starter_df = merged_df.reset_index()[["Customer Size","Num Outages"]]
obs_df = starter_df.groupby("Customer Size").mean()
#display(obs_df)
obs_stat = (obs_df.iloc[0] - obs_df.iloc[1])[0]
obs_stat 

#make an empty array for 10,000 null test stats 
results = []

for i in range(10000):
    #display(starter_df)
    test_df = starter_df.copy(deep=True)
    
    #shuffle the customer size column 
    shuffled_sizes = (
    test_df['Customer Size']
    .sample(frac=1)
    .reset_index(drop=True) 
)
    #print(shuffled_sizes)
    
    #replace the og customer size with it 
    test_df["Customer Size"] = shuffled_sizes
    
    
    #compute test stat 
    test_df = test_df.groupby("Customer Size").mean()
    #display(test_df)
    test_stat = test_df.iloc[0] - test_df.iloc[1]
    results.append(test_stat[0])


# The following code chunk plots the null test statistic and the observed test statistic.

# In[32]:


title = 'Mean Differences in Outages with States with Large and Small Total Customers'
pd.Series(results).plot(kind='hist', density=True, ec='w', bins=10, title=title)
plt.axvline(x=obs_stat, color='red', linewidth=3);

#FIGURE OUT HOW TO EXPAND Y AXIS


# By examining this plot, is it clear that the observed difference is larger than nearly all test statistics generated by the null distributions. 

# In[33]:


#calculate p value 
p_val = np.mean(results >= obs_stat)
p_val


# ### Conclusion: Do states with higher total customers have more outages?

# Using a significance level of .01 and our p value of 0, we can reasonably reject the null hypothesis in favor of the alternative hypothesis that states with larger total customers tend to have more outages. 

# ### HT2: Are outages more common in the summer months?
# 
# **Null Hypothesis**: There is no relationship between an outage occuring and what month it is. 
# 
# **Alternative Hypothesis**: Outages are more likely to occur during summer (May to August) versus other seasons. 
# 
# Test Statistic: Signed mean difference between number of outages in summer and number of outages in other seasons.  
# 
# Significance Level: 0.01. 
# 
# _Note: Summer is defined as months 5-8._

# In this following chunk, columns are added to the dataframe indicating the season and number of outages. The observed test statistic is calculated and 10000 null test statistics are also created.

# In[34]:


#make copy of dataset 
q2 = df.copy(deep = True)

#create a column for summer months 
# q2['Season'] = 'Not Summer'
# q2.loc[(q2["MONTH"]>=5) & (q2["MONTH"]<=8), "Season"] = "Summer"
q2 = q2[["MONTH", "CAUSE.CATEGORY"]]

q2 = q2.groupby("MONTH").agg(lambda x: len(list(x)))
q2 = q2.rename(columns = {"CAUSE.CATEGORY":"Num Outages"}).reset_index()
#display(q2)


q2['Season'] = 'Not Summer'
q2.loc[(q2["MONTH"]>=5) & (q2["MONTH"]<=8), "Season"] = "Summer"
starter_q2_df = q2[["Season","Num Outages"]]
#display(starter_q2_df)

obs_df = starter_q2_df.groupby("Season").mean()
obs_stat = (obs_df.iloc[0] - obs_df.iloc[1])[0]
obs_stat

results = []
for i in range(10000):
    #make new df 
    copy_df = starter_q2_df.copy(deep=True)
    
    #shuffle season column 
    shuffled_seasons = (
    copy_df['Season']
    .sample(frac=1)
    .reset_index(drop=True) 
)
    
    #add column to season 
    copy_df['Season'] = shuffled_seasons
    
    #compute test stat 
    null_df = copy_df.groupby("Season").mean()
    null_stat = (null_df.iloc[0] - null_df.iloc[1])[0]
    
    results.append(null_stat)


# The following code chunk plots the null test statistic and the observed test statistic.

# In[35]:


title = 'Mean Null Outages by Month'
pd.Series(results).plot(kind='hist', density=True, ec='w', bins=10, title=title)
plt.axvline(x=obs_stat, color='red', linewidth=3);


# In[36]:


p_val = np.mean(results <= obs_stat)
p_val


# ### Conclusion: Are outages more common in the summer months?

# Using a significance level of .01 and our p value of 0.0075, we can reasonably reject the null hypothesis in favor of the alternative hypothesis that outages are more likely to occur during summer monthes. This information can help governments and energy provides take additional preventative measures during summer.

# ### HT3: Are the causes of outages in summer different from the causes of outages year around?
# 
# **Null Hypothesis**: The causes of outages in summer are not different from the causes of outages in seasons that are not summer. 
# 
# **Alternative Hypothesis**: The causes of outages in summer are different from the causes of outages in seasons that are not summer. 
# 
# Test Statistic: TVD
# 
# Significance Level: 0.01. 
# 
# _Note: Summer is defined as months 5-8._

# This chunk creates a normalized pivot table of the number of outages by a particular cause and season.

# In[37]:


#add a column for summer 
q3 = df.copy(deep=True)
q3["Summer"] = False
q3.loc[(q3["MONTH"]>=5)&(q3["MONTH"]<=8), "Summer"] = True
q3 = q3[["Summer", "CAUSE.CATEGORY"]]
q3_pivot = q3.pivot_table(index="CAUSE.CATEGORY", columns="Summer", aggfunc='size')
#normalize the proportions
q3

cond_distr = q3_pivot / q3_pivot.sum()
cond_distr


# The following code chunk plots a horizontal bar graph by cause, season, and proportion of outages.

# In[38]:


#this line and format was borrowed from lecture
cond_distr.plot(kind='barh', title='Distribution of Cause of Outage, Conditional on Summer');


# This is the observed tvd from our given data.

# In[39]:


obs_tvd = cond_distr.diff(axis=1).iloc[:, -1].abs().sum() / 2


# This function takes in a dataframe and calculates the tvd between groups.

# In[40]:


#This chunk to calculate the tvd was barrowed from lecture
def tvd_of_groups(df):
    cnts = df.pivot_table(index="CAUSE.CATEGORY", columns="Summer", aggfunc='size')
    distr = cnts / cnts.sum()   
    return distr.diff(axis=1).iloc[:, -1].abs().sum() / 2  


# This chunk will run 10000 permutation tests and get 10000 tvds from null distributions.

# In[41]:


tvds = []
for i in range(10000):
    run_df = q3.copy(deep=True)
    shuffled_summer = run_df["Summer"].sample(frac=1).reset_index(drop=True)
    shuffled_df = run_df.loc[:, ['CAUSE.CATEGORY']].assign(Summer=shuffled_summer)
    tvds.append(tvd_of_groups(shuffled_df))

tvds = pd.Series(tvds)


# This chunk will plot the null tvds and the observed tvd.

# In[42]:


pval = (tvds >= obs_tvd).sum() / 10000
tvds.plot(kind='hist', density=True, ec='w', bins=20, title='TVDs from Null Dist', label='Simulated TVDs')
plt.axvline(x=obs_tvd, color='red', linewidth=3)
plt.legend();


# In[43]:


pval = (tvds >= obs_tvd).sum() / 10000
pval


# ### Conclusion: Are the causes of outages in summer different from the causes of outages year around?

# Using a significance level of .01 and our p value of 0.0003, we can reasonably reject the null hypothesis in favor of the alternative hypothesis that the causes of outages in summer are different from the causes of outages in seasons that are not summer.
