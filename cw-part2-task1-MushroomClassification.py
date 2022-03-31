#!/usr/bin/env python
# coding: utf-8

# In[78]:


# import the Libraries

import numpy as np        # used for multidimensional array
import pandas as pd       # used for import the dataset
import seaborn as sns
import matplotlib.pyplot as plt   # used for plotting the Graph

get_ipython().run_line_magic('matplotlib', 'inline')


# In[79]:


# import the dataset
dataset= pd.read_csv('MushroomClassification.csv')


# In[80]:


#Data understanding - first few rows of data in dataframe
dataset.head()


# In[81]:


#Data understanding - descibe dataframe - descriptive Statistics
dataset.describe()


# In[82]:


#Data understanding - data types 
dataset.dtypes


# In[83]:


#print the attributes 
dataset.keys()


# In[84]:


#data shape 
dataset.shape


# In[85]:


#Display data Information
dataset.info()


# In[86]:


#Display Data Dimensionality
dataset.ndim


# In[87]:


#you can only view data type with this function
dataset.dtypes


# In[88]:


#you can view and edit data type with this function
dataset.astype


# In[89]:


#Let us check if there is any null values
dataset.isnull().sum()


# In[90]:


dataset['class'].unique()

#Thus we have two claasification. Either the mushroom is poisonous or edible


# In[91]:


#Count of the unique occurrences of ‘class’ column
dataset['class'].value_counts()

#there are 4208 occurrences of edible mushrooms and 3916 occurrences of poisonous mushrooms


# In[135]:


#visualize the count of edible and poisonous mushrooms using Seaborn

count = dataset['class'].value_counts()
plt.figure(figsize=(8,7))

sns.barplot(count.index, count.values, alpha=0.8, palette="prism")

plt.ylabel('Count', fontsize=12)
plt.xlabel('Class', fontsize=12)

plt.title('Number of poisonous/edible mushrooms')

plt.show()

#Using value_counts() method we can see that the dataset is balanced


# In[93]:


#the dataset has values in strings.
#We need to convert all the unique values to integers

from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
for col in dataset.columns:
    dataset[col] = labelencoder.fit_transform(dataset[col])


# In[94]:


#Print the new dataset
dataset.head()


# In[100]:


#Checking the encoded values
dataset['stalk-color-above-ring'].unique()


# In[96]:


dataset['veil-type']


# In[97]:


#the column “veil-type” is 0 and not contributing to the data so we’ll remove it.

dataset = dataset.drop(["veil-type"],axis=1)


# In[98]:


#Print the new dataset without the veil type
dataset.head()


# In[28]:


#print the size of the dataset 
print(dataset.groupby('class').size())


# In[175]:


#quick look of the charateristics of the data

dataset_div = pd.melt(dataset, "class", var_name="Characteristics")
fig, ax = plt.subplots(figsize=(16,6))

p = sns.violinplot(ax = ax, x="Characteristics", y="value", hue="class", split = True, data=dataset_div, inner = 'quartile', palette = 'Set1')

dataset_no_class = dataset.drop(["class"],axis = 1)

p.set_xticklabels(rotation = 90, labels = list(dataset_no_class.columns));


# In[ ]:





# In[102]:


#correlation between the variables - using a heat map 

plt.figure(figsize=(14,12))
sns.heatmap(dataset.corr(),linewidths=.1,cmap="Purples", annot=True, annot_kws={"size": 7})
plt.yticks(rotation=0);


# In[35]:


# Create a figure instance
fig, axes = plt.subplots(nrows=2 ,ncols=2 ,figsize=(9, 9))

# Create an axes instance and the boxplot
bp1 = axes[0,0].boxplot(dataset['stalk-color-above-ring'],patch_artist=True)

bp2 = axes[0,1].boxplot(dataset['stalk-color-below-ring'],patch_artist=True)

bp3 = axes[1,0].boxplot(dataset['stalk-surface-below-ring'],patch_artist=True)

bp4 = axes[1,1].boxplot(dataset['stalk-surface-above-ring'],patch_artist=True)

#Plotting boxplot to see the distribution of the data


# In[38]:


#Separating features and label

X = dataset.iloc[:,1:23]  # all rows, all the features and no labels
y = dataset.iloc[:, 0]  # all rows, label only

X.head()
y.head()


# In[39]:


X.describe()


# In[40]:


y.head()


# In[42]:


dataset.corr()


# In[43]:


#Standardising the data
# Scale the data to be between -1 and 1

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X=scaler.fit_transform(X)

X


# In[44]:


#Principal Component Analysis

from sklearn.decomposition import PCA
pca = PCA()
pca.fit_transform(X)


# In[45]:


#covariance
covariance=pca.get_covariance()


# In[46]:


explained_variance=pca.explained_variance_
explained_variance


# In[47]:


with plt.style.context('dark_background'):
    plt.figure(figsize=(6, 4))
    
    plt.bar(range(22), explained_variance, alpha=0.5, align='center',
            label='individual explained variance')
    
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    
    plt.tight_layout()
    
    #We can see that the last 4 components has less amount of variance of the data.
    #The 1st 17 components retains more than 90% of the data.


# In[49]:


#Let us take only first two principal components and visualise it using K-means clustering
N=dataset.values

pca = PCA(n_components=2)
x = pca.fit_transform(N)

plt.figure(figsize = (5,5))
plt.scatter(x[:,0],x[:,1])
plt.show()


# In[51]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=5)
X_clustered = kmeans.fit_predict(N)

LABEL_COLOR_MAP = {0 : 'g',
                   1 : 'y'
                  }

label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]
plt.figure(figsize = (5,5))
plt.scatter(x[:,0],x[:,1], c= label_color)
plt.show()

#Thus using K-means we are able segregate 2 classes well using the first two components with maximum variance.


# In[54]:


#Splitting the data into training and testing dataset

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=4)


# In[55]:


#Default Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics

model_LR= LogisticRegression()


# In[56]:


model_LR.fit(X_train,y_train)


# In[57]:


y_prob = model_LR.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  
y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
model_LR.score(X_test, y_pred)


# In[58]:


confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
confusion_matrix


# In[59]:


auc_roc=metrics.roc_auc_score(y_test,y_pred)
auc_roc


# In[60]:


from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc


# In[61]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],linestyle='--')

plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# In[108]:


#Define the predictor and target attributes

X = dataset.iloc[:,:-1].values    # predictor attribute
y = dataset.iloc[:,-1].values        # target attribute


# In[109]:


#view the predictor attribute 
dataset.iloc[:,:-1].values    # predictor attribute


# In[110]:


#view the target attribute 
dataset.iloc[:,-1].values        # target attribute


# In[163]:


# split the dataset into test set and train set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =train_test_split(X,y, test_size=0.25, random_state=0)

#A 75% dataset split for training and the remaining 25% for testing.


# In[164]:


#preparing the data 

from sklearn.model_selection import train_test_split

# "class" column as numpy array.
y = dataset["class"].values

# All data except "class" column.
x = dataset.drop(["class"], axis=1).values

# Split data for train and test.
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=42,test_size=0.2)


# In[165]:


#Logistic Regression Classification

from sklearn.linear_model import LogisticRegression

## lr = LogisticRegression(solver="lbfgs")
lr = LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)

print("Test Accuracy: {}%".format(round(lr.score(x_test,y_test)*100,2)))


# In[166]:


from sklearn.neighbors import KNeighborsClassifier

best_Kvalue = 0
best_score = 0

for i in range(1,10):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    if knn.score(x_test,y_test) > best_score:
        best_score = knn.score(x_train,y_train)
        best_Kvalue = i

print("""Best KNN Value: {}
Test Accuracy: {}%""".format(best_Kvalue, round(best_score*100,2)))


# In[167]:


from sklearn.svm import SVC

svm = SVC(random_state=42, gamma="auto")
svm.fit(x_train,y_train)

print("Test Accuracy: {}%".format(round(svm.score(x_test,y_test)*100,2)))


# In[168]:


from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(x_train,y_train)

print("Test Accuracy: {}%".format(round(nb.score(x_test,y_test)*100,2)))


# In[177]:


dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
print("Test Accuracy: {}%".format(round(dt.score(x_test,y_test)*100,2)))


# In[170]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train,y_train)

print("Test Accuracy: {}%".format(round(rf.score(x_test,y_test)*100,2)))


# In[171]:


from sklearn.metrics import confusion_matrix

# Linear Regression
y_pred_lr = lr.predict(x_test)
y_true_lr = y_test

cm = confusion_matrix(y_true_lr, y_pred_lr)
f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred_lr")
plt.ylabel("y_true_lr")
plt.show()


# In[172]:


# Random Forest
y_pred_rf = rf.predict(x_test)
y_true_rf = y_test

cm = confusion_matrix(y_true_rf, y_pred_rf)
f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred_rf")
plt.ylabel("y_true_rf")
plt.show()


# In[ ]:


#---- conclusion -----

#From the confusion matrix, we saw that our train and test data is balanced.

#Most of classfication methods hit 100% accuracy with this dataset.

