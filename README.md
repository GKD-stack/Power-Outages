# Causes of Power Outages

### Introduction
The prediction problem I am attempting is "predict the cause of a major power outage." This is a classification problem. I am attempting to understand what was the main cause of a particular power outage using the information provided so the target variable is "CAUSE.CATEGORY". I evaluate the following models with the F1 score because because it achieves a balance between percision and recall while accounting for the imbalance in cause category (which accuracy doesn't). 

### Data Analaysis: Where and when do major power outages tend to occur? (Exploratory data analysis, missingness analyses, hypothesis tests) 
The data is downloadable [here](https://engineering.purdue.edu/LASCI/research-data/outages/outagerisks).
data dictionary: [article](https://www.sciencedirect.com/science/article/pii/S2352340918307182) 

### Baseline Model
**Model:** Decision Tree Classifier.  
**7 Features:**

    - Year (Ordinal)
    - Month (Ordinal)
    - US State (Nominal)
    - Climate Region (Nominal)
    - Population (Quantitative)
    - Percent Water Inland (Quantitative)
    - Area Percent Urban (Quantitative) 
**Performance:**
I used the F1 score as the primary metric for classification. I didn't deem percision or recall as being more relevant in this particular context so I picked the F1 score as a harmonic mean. I didn't choose accuracy because of the imbalances in the causes of power outages. 

I also broke down F1-score by weighted, micro, and macro to provide additional insight. For this particular case, I believe that the macro F1 score is most relevant because it weighs each class equally, directly addressing the class imbalance whereas micro weighs each observation equally, which poses the same problem as using accuracy. 

    - F1 score (macro): 0.35309892523229475
    - F1 score (micro): 0.5859375
    - F1 score (weighted): 0.5788115441845282

Since the higher the F1 score is the better, my model is currently ok according to (https://stephenallwright.com/good-f1-score/). However, it must be compared to other models trained and evaluated on the same tests for a more comprehensive evaluation. 

### Final Model
**Features:**
These are the features I engineered in my model:  
- **Engineered:**
    - Month: Based off of project 3, it was clear there were more outages during summer monthes. Therefore, I used a function transformer to transform this variable into a 1 if it was between 5 an 8 months and 0 otherwise. 
    - Urban: Based off of project 3, there was a positive correlation between larger states on the coast and the number of outages. Therefore, if a state had an urban area percentage that was above average in this dataset, I used a Binarizer to give it a 1, otherwise it was a 0.  
- **One Hot Encoded:**
    - Year: I included this a categorical variable because during certain years (based off of project 3), there were a series of events (ie natural disasters) that caused significantly more outages during certain years than others. 
    - US State: I included this because certain states experienced power outages significantly more than others due to factors such as their size, customers, and geographic location. 
    - NERC Region: I included this because the NERC region depends on the geographic area of a state which is highly correlated to the number and type of outage that can occur within an area. 
    - Climate Region: I included this because the climate of a certain location has a correlation to the type of outage that will occur. Recent emperical examples of Texas and Florida further support this. 
- **Passed Through:**
    - Total Sales: I included this because this variable reveals the total electrical consumption in a particular state. If this consumption exceeds limits, it could cause a particular type of outage. 
    - Total Customers: I included this because as this number rises, it imposes additional pressure on an electrical grid which could cause a particular type of outage. 
    - Population: I included this because based off of project 3, it was clear that larger (more population) states had more outages likely due to more demand which could cause a particular type of outage. 
    - Urban Cluster Area: I included this for the same reason I engineered the Urban variable as described above. 
    - Percent Land: I included this for a similar reason as population. Although its clear that larger states such as California have more outages, the reason why (population, size) isn't clear. But percent land is a viable proxy for a "large" state because its still correlated with a larger population and more expansive electrical grid. 
    - Percent Water Total: I included this because the amount of water area in a state is correlated to specific types of causes that depend on natural disasters. 
    - Percent Water Inland: I included this as a balancing variable against percent land because it is plausable that a state may have a lot of land but much of it may be unlivable because it is occupied by bodies of water. For example, even if a state is large, it shouldn't be more likely to have a particular type of outage caused by demand because it also has a high percent of water inland. 
    
**Model Selection**

- **Decision Tree**
    - **Best Parameters**
        - {'tree__criterion': 'gini','tree__max_depth': 18,'tree__min_samples_split': 3}
    - **Evaluation Metrics**
        - F1 score (macro): 0.37658562670222623
        - F1 score (micro): 0.6558441558441559
        - F1 score (weighted): 0.6426739247522872

- **Random Forest**
    - **Best Parameters**
        - {'rf__criterion': 'gini', 'rf__max_depth': 10, 'rf__min_samples_split': 2, 'rf__n_estimators': 120}
    - **Evaluation Metrics** 
        - F1 score (macro): 0.3087639653562575
        - F1 score (micro): 0.7142857142857143
        - F1 score (weighted): 0.65112615579147020.6511261557914702
        
- **KNN**
    - **Best Parameters**
        - {'knn__algorithm': 'ball_tree', 'knn__n_neighbors': 19,     'knn__weights': 'distance'}
    - **Evaluation Metrics**
        - F1 score (macro): 0.39082644712896814
        - F1 score (micro): 0.6948051948051948
        - F1 score (weighted): 0.6637773715122837

<center><img src="p5.png" width="80%"></center>

**Chosen Model**

All the models had similar F1 scores. I choose a random forest classification model because I believe this is the most accurate since it has utilized several decision trees, which would mitigate bias and error that could arise during the training process. Furthermore, alongside its accuracy, the random forest can still capture non linear relationships, which is ideal for phenomana such as weather. 


### Fairness Evaluation
**Parity Measure:** False Negative Rate. I chose this to understand whether the predicted cause category for a state is the same for Democratic states and Republican states. 

**Null Hypothesis:** My model is fair. The false negative rate for states with a Republican control as the same as states with a Democratic control. 

**Alternative Hypothesis:** My model is unfair. The false negative rate for Republican controlled states is higher.

**Justification of Measure and Hypothesis**: I believe that states with a Republican control are more likely to have a false negative rate  because they are less likely oriented towards data collection efforts due to their (and their voting bases') lack of emphasis on using scientific techniques as preventative measures. So Democrat controlled states are also more likely to provide data and hence I believe a higher proportion of particular causes will be correctly classified so they will have a lower false negative rate.

I did not select demographic parity because we cannot assume that the proportion of times the classifier predicts the cause correctly is independent of political control because Democratic states geographic locations are correlated with cause as learned through project three.

**Political Affiliation Data**: I found a dataset from https://worldpopulationreview.com/state-rankings/states-by-political-party that highlights each states political affiliation through data that includes "legislative majority paired with governor control (as seen in the table above), party affiliation of each state's governor, senate, and house, and 
percentage of adults who identify as a democrat, republican, or neither".

**Results**: With a p value of .147 through 1000 iterations in a permutation test, I fail to reject the null hypothesis that my model is fair for Republican states and Democratic states using a parity measure of false negative rate. 
