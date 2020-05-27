#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import scipy.stats as stats
import pandas_profiling


plt.rcParams['figure.figsize'] = 10, 7.5
plt.rcParams['axes.grid'] = True

from matplotlib.backends.backend_pdf import PdfPages


# In[6]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# In[7]:


#Load the dataset

credit = pd.read_csv("C:/Data Sets/(Class 21-Python - Case Study)/4. Segmentation of Credit Card Customers/CC_GENERAL.csv")


# In[8]:


#Data Inspection
credit.info()


# In[5]:


#Detailed profiling using pandas profiling

pandas_profiling.ProfileReport(credit)


# In[6]:


#Exporting pandas profiling output to html file

output = pandas_profiling.ProfileReport(credit)

output.to_file('C:/Data Sets/(Class 21-Python - Case Study)/4. Segmentation of Credit Card Customers/pandas_profiling.html')


# In[9]:


#Total no. of columns in the dataset
len(credit.columns)


# In[10]:


#Separating Categorical and Numerical variables
numeric_var_names=[key for key in dict(credit.dtypes) if dict(credit.dtypes)[key] in ['float64', 'int64', 'float32', 'int32']]
cat_var_names=[key for key in dict(credit.dtypes) if dict(credit.dtypes)[key] in ['object']]
print(numeric_var_names)
print(cat_var_names)


# In[11]:


len(credit.columns)
len(numeric_var_names)

#All variables are numeric except CUST_ID which is Categorical.
#We will disregard CUST_ID anyways as it is not of significance.


# In[12]:


credit_cat = credit[cat_var_names]
credit_cat.head(5)

credit_num=credit[numeric_var_names]
credit_num.head(5)


# In[13]:


# Creating Data audit Report
# Use a general function that returns multiple values
def var_summary(x):
    return pd.Series([x.count(), x.isnull().sum(), x.sum(), x.mean(), x.median(),  x.std(), x.var(), x.min(), x.dropna().quantile(0.01), 
                      x.dropna().quantile(0.05),x.dropna().quantile(0.10),x.dropna().quantile(0.25),x.dropna().quantile(0.50),x.dropna().quantile(0.75), 
                      x.dropna().quantile(0.90),x.dropna().quantile(0.95), x.dropna().quantile(0.99),x.max()], 
                  index=['N', 'NMISS', 'SUM', 'MEAN','MEDIAN', 'STD', 'VAR', 'MIN', 'P1' , 'P5' ,'P10' ,'P25' ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX'])

num_summary=credit_num.apply(var_summary)


# In[14]:


credit_num.apply(var_summary).round(2)

#We have outliers in our data and missing values in some columns.


# In[15]:


#Handling Outliers - at 99%tile or 95%tile if required after including some particular vars like income
def outlier_capping(x):
    x = x.clip(upper=x.quantile(0.99))
    x = x.clip(lower=x.quantile(0.01))
    return x

credit_num=credit_num.apply(lambda x: outlier_capping(x))


# In[16]:


#Handling missings - Method2
def Missing_imputation(x):
    x = x.fillna(x.mean())
    return x

credit_num=credit_num.apply(lambda x: Missing_imputation(x))


# In[17]:


#Checking the corelation between the data and importing it to excel for analysis
credit_corr = credit_num.corr()
credit_corr.to_excel('C:/Data Sets/(Class 21-Python - Case Study)/4. Segmentation of Credit Card Customers/corr.xlsx')


# In[20]:


plt.subplots(figsize=(20,15))
sns.heatmap(credit_num.corr(),cmap = 'viridis', annot = True)
plt.show()


# In[21]:


#Monthly average purchase and cash advance amount
credit_num.PURCHASES.mean()


# In[22]:


#Purchases by type (one-off, installments)
credit_num.ONEOFF_PURCHASES.mean()
credit_num.INSTALLMENTS_PURCHASES.mean()


# In[23]:


#Limit usage (balance to credit limit ratio)

credit_num.BALANCE/credit_num.CREDIT_LIMIT


# In[24]:


#Payments to minimum payments ratio etc

credit_num.PAYMENTS/credit_num.MINIMUM_PAYMENTS


# In[25]:


credit_num.columns


# # Feature Selection
# We will talk about initial feature selection now.
# We have already removed CUST_ID as it is a categorical variable with high cardinality.
# PURCHASES and ONEOFF_PURCHASES have a very high correlation with each other and we will take a step and remove ONEOFF_PURCHASES
# as PURCHASES is a more important variable based on the context.
# Removed Tenure as it does not impact the marketing decisions of the credit card company.

# In[26]:


credit_num.drop(['ONEOFF_PURCHASES','TENURE'], axis=1, inplace=True)


# In[27]:


credit_num.head()


# In[28]:


len(credit_num.columns)


# In[29]:


#Standardizing the variables

sc = StandardScaler()
credit_scaled = sc.fit_transform(credit_num)


# In[30]:


pd.DataFrame(credit_scaled)


# In[31]:


#Summarizing the Standardizied variables
pd.DataFrame(credit_scaled).apply(var_summary).round(2)


# ## Applying PCA for variable reduction/checking multicollinearity

# In[32]:


pca = PCA(n_components = 15)


# In[33]:


pca.fit(credit_scaled)


# In[34]:


pca.explained_variance_ #Eigen value


# In[35]:


sum(pca.explained_variance_)


# In[36]:


#The amount of variance that each PC explains
var= pca.explained_variance_ratio_
var


# In[37]:


# Screen Plot - Plotting Explained Variance % by each componet
plt.figure()
plt.plot(var)
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Telco Dataset Explained Variance')
plt.show()


# In[38]:


#Cumulative Variance explains
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)


# In[39]:


pd.DataFrame({'Eigen_value': pca.explained_variance_, 'Cumm_variance':var1})


# In[40]:


#We can choose 4 or 5 components based on the Eigen values.
#We will go with 4 components as the multicollinearity is negiligble and cumm variance captures 77.40 variance of the data.


# In[41]:


pc_final = PCA(n_components = 4)
pc_final.fit(credit_scaled)


# In[42]:


pc_final.explained_variance_


# In[43]:


pc_final.explained_variance_ratio_


# In[44]:


reduced_cr=pc_final.fit_transform(credit_scaled)


# In[45]:


dimensions = pd.DataFrame(reduced_cr)


# In[46]:


dimensions.columns = ["C1", "C2", "C3", "C4"]


# In[47]:


print(credit_scaled.shape)

print(dimensions.shape)
#pd.DataFrame(telco_scaled).head()
print(dimensions.head())


# ## Clustering/Segmentation

# In[48]:


km_3=KMeans(n_clusters=3,random_state=123).fit(dimensions)
km_4=KMeans(n_clusters=4,random_state=123).fit(dimensions)
km_5=KMeans(n_clusters=5,random_state=123).fit(dimensions)
km_6=KMeans(n_clusters=6,random_state=123).fit(dimensions)
km_7=KMeans(n_clusters=7,random_state=123).fit(dimensions)
km_8=KMeans(n_clusters=8,random_state=123).fit(dimensions)


# In[49]:


credit_num['cluster_3'] = km_3.labels_
credit_num['cluster_4'] = km_4.labels_
credit_num['cluster_5'] = km_5.labels_
credit_num['cluster_6'] = km_6.labels_
credit_num['cluster_7'] = km_7.labels_
credit_num['cluster_8'] = km_8.labels_


# In[50]:


credit_num


# In[51]:


pd.Series(km_3.labels_).value_counts()/sum(pd.Series(km_3.labels_).value_counts())


# In[52]:


pd.Series(km_4.labels_).value_counts()/sum(pd.Series(km_4.labels_).value_counts())


# In[53]:


pd.Series(km_5.labels_).value_counts()/sum(pd.Series(km_5.labels_).value_counts())


# In[54]:


pd.Series(km_6.labels_).value_counts()/sum(pd.Series(km_6.labels_).value_counts())


# In[55]:


pd.Series(km_7.labels_).value_counts()/sum(pd.Series(km_7.labels_).value_counts())


# In[56]:


pd.Series(km_8.labels_).value_counts()/sum(pd.Series(km_8.labels_).value_counts())


# ## Quantitative Analysis
# Choosing number clusters using Silhouette Coefficient

# In[57]:


from sklearn import metrics
metrics.silhouette_score(dimensions, km_3.labels_)


# In[58]:


# calculate SC for K=3 through K=12
k_range = range(2, 20)
scores = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=123)
    km.fit(dimensions)
    scores.append(metrics.silhouette_score(dimensions, km.labels_))


# In[59]:


scores


# In[60]:


# plot the results
plt.plot(k_range, scores)
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Coefficient')
plt.grid(True)


# ## Note:
# 
# The solution can be 3-7 based on the SC score. If we take highest SC score, 3 segment solution is best. But 3 is highly skewed.

# ## Elbow Analysis

# In[61]:


cluster_range = range( 2, 20 )
cluster_errors = []

for num_clusters in cluster_range:
    clusters = KMeans( num_clusters )
    clusters.fit( dimensions )
    cluster_errors.append( clusters.inertia_ )


# In[62]:


clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )

clusters_df[0:10]


# In[63]:


# allow plots to appear in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )


# ## Note:
# The elbow diagram shows that the gain in explained variance reduces significantly from 3 to 4 to 5. So, optimal number of clusters could either 4 or 5.
# The actual number of clusters chosen can be finally based on business context and convenience of dealing with number of segments or clusters.

# # Qualitative Analysis with Profiling

# In[64]:


credit_num.cluster_3.value_counts()*100/sum(credit_num.cluster_3.value_counts())


# In[65]:


size=pd.concat([pd.Series(credit_num.cluster_3.size), pd.Series.sort_index(credit_num.cluster_3.value_counts()), pd.Series.sort_index(credit_num.cluster_4.value_counts()),
           pd.Series.sort_index(credit_num.cluster_5.value_counts()), pd.Series.sort_index(credit_num.cluster_6.value_counts()),
           pd.Series.sort_index(credit_num.cluster_7.value_counts()), pd.Series.sort_index(credit_num.cluster_8.value_counts())])


# In[66]:


Seg_size=pd.DataFrame(size, columns=['Seg_size'])
Seg_Pct = pd.DataFrame(size/credit_num.cluster_3.size, columns=['Seg_Pct'])


# In[67]:


pd.concat([Seg_size.T, Seg_Pct.T], axis=0)


# In[68]:


# Mean value gives a good indication of the distribution of data. So we are finding mean value for each variable for each cluster
Profling_output = pd.concat([credit_num.apply(lambda x: x.mean()).T, credit_num.groupby('cluster_3').apply(lambda x: x.mean()).T, credit_num.groupby('cluster_4').apply(lambda x: x.mean()).T,
          credit_num.groupby('cluster_5').apply(lambda x: x.mean()).T, credit_num.groupby('cluster_6').apply(lambda x: x.mean()).T,
          credit_num.groupby('cluster_7').apply(lambda x: x.mean()).T, credit_num.groupby('cluster_8').apply(lambda x: x.mean()).T], axis=1)


# In[69]:


Profling_output


# In[70]:


Profling_output_final=pd.concat([Seg_size.T, Seg_Pct.T, Profling_output], axis=0)


# In[71]:


Profling_output_final


# In[72]:


Profling_output_final.columns = ['Overall', 'KM3_1', 'KM3_2', 'KM3_3',
                                'KM4_1', 'KM4_2', 'KM4_3', 'KM4_4',
                                'KM5_1', 'KM5_2', 'KM5_3', 'KM5_4', 'KM5_5',
                                'KM6_1', 'KM6_2', 'KM6_3', 'KM6_4', 'KM6_5','KM6_6',
                                'KM7_1', 'KM7_2', 'KM7_3', 'KM7_4', 'KM7_5','KM7_6','KM7_7',
                                'KM8_1', 'KM8_2', 'KM8_3', 'KM8_4', 'KM8_5','KM8_6','KM8_7','KM8_8',]


# In[73]:


Profling_output_final


# In[74]:


Profling_output_final.to_csv('C:/Data Sets/(Class 21-Python - Case Study)/1. Predicting Credit Card Spend & Identifying Key Drivers/Profiling_output.csv')


# Conclusion: Conclusion: We will not go with 3 and 4 clusters because one cluster is capturing more than 40% data which will generalize the clusters.																			
# We will go with 6 clusters because the one uncommon cluster in 6 clusters is capturing relevant data and is new information.											
# Therefore, we will go with 6 clusters as per the qualitative analysis for clustering in which all clusters have specefic properties.											
# 
# 
# 
# Kindly check the Profiling_output excel file for qualitative analysis.	
# 
# If there are any areas I can improve, kindly let me know about those.
# 
# THANK YOU!
# 

# In[ ]:




