#!/usr/bin/env python
# coding: utf-8


# In[1]:



# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go

# Load libraries
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

## Following are for plotting the Decison Tree Diagram
from sklearn.tree import export_graphviz
#from sklearn.externals.six import StringIO
from six import StringIO  
from IPython.display import Image  
#import pydotplus  # commenting out the plotting due to Erdos limitations
from sklearn.ensemble import VotingClassifier

# classification report for our models
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# allowing for any single variable to print out without using the print statement:
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import random
print ('Sucessfully imported the required libraries.')


# In[3]:


train_data=pd.read_csv('clean_census.csv')
train_data.head()
train_data.shape


# In[4]:


test_data=pd.read_csv('clean_test.csv')
test_data.head()
test_data.shape


# In[5]:


train_labels = train_data.pop('class')
test_labels = test_data.pop('class')

train_labels.shape
test_labels.shape


# In[6]:


## Create KNN Classifier by Varying k= 1 to 51
best_test_accuracy, corresponding_acc_on_trn, k_for_best_accuracy = 0, 0, 0
#knn_range = range(1,51,20)
for k in range(1,51,10):
    knn = KNeighborsClassifier(n_neighbors=k)

    #Train the model using the training sets
    knn.fit(train_data, train_labels)

    #Predict the response for test dataset
    train_pred = knn.predict(train_data)
    # Model Accuracy for KNN for K=5, how often is the classifier correct?
    train_accuracy=metrics.accuracy_score(train_labels, train_pred)
    #print("Train Data accuracy for KNN based Classifier for K = ", k, "  is ",train_accuracy)
    
    #Predict the response for test dataset
    test_pred = knn.predict(test_data)
    # Model Accuracy for KNN for K=5, how often is the classifier correct?
    test_accuracy=metrics.accuracy_score(test_labels, test_pred)
    print("Test Data accuracy for KNN based Classifier for K = ", k, "  is ",test_accuracy)
    
    if(best_test_accuracy < test_accuracy):
        #print("Replacing the best model")
        best_test_accuracy = test_accuracy
        corresponding_acc_on_trn = train_accuracy
        k_for_best_accuracy = k
        knn_best=knn

print("\n\nBest Accuracy for KNN based Classifier on test data K = ", k_for_best_accuracy, "  is ",best_test_accuracy, end =" ")
print(", corresponding accuracy on train data: ",corresponding_acc_on_trn)


# In[7]:


random.seed(11)
best_test_acc_dt, corr_acc_on_trn_dt, depth_of_best_acc = 0, 0, 0
for max_depth in range(1,51,2):
    # Create Decision Tree classifer object
    dtc = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
    # Train Decision Tree Classifer
    dtc = dtc.fit(train_data,train_labels)
    
    #Predict the response for train dataset
    train_pred = dtc.predict(train_data)
    # Model Accuracy for Decision Tree, how often is the classifier correct?
    train_accuracy=metrics.accuracy_score(train_labels, train_pred)
    #print("Train Data accuracy for Decision Tree Model based Classification for maximum depth of ", max_depth, " is ",train_accuracy)
    
    #Predict the response for test dataset
    test_pred = dtc.predict(test_data)
    # Model Accuracy for Decision Tree, how often is the classifier correct?
    test_accuracy=metrics.accuracy_score(test_labels, test_pred)
    #print("Test Data accuracy for Decision Tree Model based Classification for maximum depth of ", max_depth, " is ",test_accuracy)    

    if(best_test_acc_dt < test_accuracy):
        #print("Replacing the best model")
        best_test_acc_dt = test_accuracy
        corresponding_acc_on_trn = train_accuracy
        depth_of_best_acc  = max_depth
        dtc_best=dtc
        
print("\n\nBest Test Data Accuracy for Decision Tree Model based Classification for maximum depth of ", end =" ")
print(depth_of_best_acc, "  is ",best_test_acc_dt, end =" ")
print(", corresponding accuracy on train data: ",corresponding_acc_on_trn)


random.seed(10)
best_test_acc_rf, corr_acc_on_trn_rf, depth_of_best_acc_rf, n_est_4r_best_acc = 0, 0, 0, 0
for max_depth in range(1,51,10):
    for n_estimators in range(1,201,25):
        #Create a Random Forest based Classifier
        rnfc=RandomForestClassifier(criterion="entropy", max_depth=max_depth,n_estimators=n_estimators)

        #Train the model using the training sets y_pred=clf.predict(X_test)
        rnfc.fit(train_data,train_labels)

        train_pred=rnfc.predict(train_data)
        # Model Accuracy for Random Forest, how often is the classifier correct?
        train_accuracy=metrics.accuracy_score(train_labels, train_pred)
        print("Train Data accuracy for Random Forest with maximum depth ",max_depth," and n_estimators ", n_estimators, " is " ,train_accuracy)
        
        
        test_pred=rnfc.predict(test_data)
        # Model Accuracy for Random Forest, how often is the classifier correct?
        test_accuracy=metrics.accuracy_score(test_labels, test_pred)
        print("Test Data accuracy for Random Forest with maximum depth ",max_depth," and n_estimators ", n_estimators, " is " ,test_accuracy)
        
        if(best_test_acc_rf < test_accuracy):
            print("Replacing the best model")
            best_test_acc_rf=test_accuracy
            corr_acc_on_trn_rf = train_accuracy
            depth_of_best_acc_rf = max_depth
            n_est_4r_best_acc = n_estimators
            rnfc_best=rnfc
            
print("\n\nBest Test Data Accuracy for Random Forest Model with maximum depth of ", depth_of_best_acc_rf,  end =" ")       
print(" n_estimators of ",n_est_4r_best_acc , "  is ",best_test_acc_rf, end =" ")
print(", corresponding accuracy on train data: ",corresponding_acc_on_trn)


# In[11]:


## Now create Naïve Bayes classifier
#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb = gnb.fit(train_data,train_labels)

#Predict the response for test dataset
train_pred = gnb.predict(train_data)
train_acc_gnb = metrics.accuracy_score(train_labels, train_pred)

#Predict the response for test dataset
test_pred = gnb.predict(test_data)
# Model Accuracy for Random Forest, how often is the classifier correct?
test_acc_gnb = metrics.accuracy_score(test_labels, test_pred)

print("Train Data accuracy for Gaussian Naive Bayes: ",train_acc_gnb)
print("Test Data accuracy for Gaussian Naive Bayes: ",test_acc_gnb)


# In[12]:


# passing our previous models to our ensemble model:

## remove ('Naive Bayes', gnb)
voting_clf_hard = VotingClassifier(
    estimators=[('knn', knn_best), ('Decision_tree', dtc_best), ('Random_Forest', rnfc_best)], voting='hard', flatten_transform=True)


# fit the ensemble
voting_clf_hard.fit(train_data, train_labels)

# predicting the ensemble
train_pred_hard = voting_clf_hard.predict(train_data)
train_pred_acc = metrics.accuracy_score(train_labels, train_pred_hard) 
print("Train Data Ensemble Accuracy, Hard Voting: ", train_pred_acc)

# predicting the ensemble
test_pred_hard = voting_clf_hard.predict(test_data)
test_pred_acc = metrics.accuracy_score(test_labels, test_pred_hard) 
print("Test Data Ensemble Accuracy, Hard Voting: ",test_pred_acc)


# In[13]:

# remove ('Naive Bayes', gnb)
voting_clf_soft = VotingClassifier(estimators=[('knn', knn_best), ('Decision_tree', dtc_best), ('Random_Forest', rnfc_best)], voting='soft', flatten_transform=True)
  
# fit the ensemble
voting_clf_soft.fit(train_data, train_labels)

# predicting the ensemble
train_pred_soft = voting_clf_soft.predict(train_data)
train_pred_acc = metrics.accuracy_score(train_labels, train_pred_soft) 
print("Train Data Ensemble Accuracy, Soft Voting: ", train_pred_acc)

# predicting the ensemble
test_pred_soft = voting_clf_soft.predict(test_data)
test_pred_acc = metrics.accuracy_score(test_labels, test_pred_soft) 
print("Test Data Ensemble Accuracy, Soft Voting: ",test_pred_acc)


# In[14]:


# createing a dataframe to store all the accuracy, so that in future, it become easier to plot the results
# data_type : 'train' or 'test'
# algorithm_name: 'knn', 'decision tree', 'random forest', 'bayes', 'hard ensemble', 'soft ensemble'
# accuracy in float beween 0 to 100

results_df = pd.DataFrame(columns=['data_type', 'algorithm_name', 'accuracy'])
results_df


# In[15]:


# KNN Accuracy and Classification Report
train_pred = knn_best.predict(train_data)
train_knn_accuracy=metrics.accuracy_score(train_labels, train_pred)
print("Train Data Accuracy for KNN algorithm is ", train_knn_accuracy)
print("Classification report: ")
train_cls_knn_report=classification_report(train_labels, train_pred)
print(train_cls_knn_report)

#  'data_type', 'algorithm_name', 'accuracy'
results_df = results_df.append({'data_type': 'train', 'algorithm_name': 'knn', 'accuracy': train_knn_accuracy * 100}, ignore_index=True)

# KNN Accuracy and Classification Report
test_pred = knn_best.predict(test_data)
test_knn_accuracy=metrics.accuracy_score(test_labels, test_pred)
print("Test Data KNN Accuracy and Classification Report: ", test_knn_accuracy)
test_cls_knn_report=classification_report(test_labels, test_pred)
print(test_cls_knn_report)

#  'data_type', 'algorithm_name', 'accuracy'
results_df = results_df.append({'data_type': 'test', 'algorithm_name': 'knn', 'accuracy': test_knn_accuracy * 100}, ignore_index=True)


# In[ ]:





# In[16]:


# Decision Accuracy and Classification Report for Decision Tree
train_pred = dtc_best.predict(train_data)
train_dt_accuracy=metrics.accuracy_score(train_labels, train_pred)

print("Train Data Decision Tree Accuracy : ", train_knn_accuracy)
print(classification_report(train_labels, train_pred))

#  'data_type', 'algorithm_name', 'accuracy'
results_df = results_df.append({'data_type': 'train', 'algorithm_name': 'decision_tree', 'accuracy': train_dt_accuracy * 100}, ignore_index=True)


# Decision Accuracy and Classification Report for Decision Tree
test_pred = dtc_best.predict(test_data)
test_dt_accuracy=metrics.accuracy_score(test_labels, test_pred)
print("Test Data Decision Tree Accuracy: ", test_dt_accuracy)
print(classification_report(test_labels, test_pred))

#  'data_type', 'algorithm_name', 'accuracy'
results_df = results_df.append({'data_type': 'test', 'algorithm_name': 'decision_tree', 'accuracy': test_dt_accuracy * 100}, ignore_index=True)


# In[17]:


# Classification Report for Random Forest
train_pred = rnfc_best.predict(train_data)
train_rf_accuracy=metrics.accuracy_score(train_labels, train_pred)

print("Train Data Random Forest Accuracy and Classification Report: ", train_rf_accuracy)
print(classification_report(train_labels, train_pred))

#  'data_type', 'algorithm_name', 'accuracy'
results_df = results_df.append({'data_type': 'train', 'algorithm_name': 'random_forest', 'accuracy': train_rf_accuracy * 100}, ignore_index=True)


# Classification Report for random forest
test_pred = rnfc_best.predict(test_data)
test_rf_accuracy=metrics.accuracy_score(test_labels, test_pred)

print("Test Data Random Forest Accuracy and Classification Report: ", test_rf_accuracy)
print(classification_report(test_labels, test_pred))

#  'data_type', 'algorithm_name', 'accuracy'
results_df = results_df.append({'data_type': 'test', 'algorithm_name': 'random_forest', 'accuracy': test_rf_accuracy * 100}, ignore_index=True)


# In[18]:


# Decision Accuracy and Classification Report for Naïve Bayes classifier
train_pred = gnb.predict(train_data)
train_nb_accuracy=metrics.accuracy_score(train_labels, train_pred)

print("Train Data Naive Bayes Accuracy : ", train_nb_accuracy)
print(classification_report(train_labels, train_pred))


#  'data_type', 'algorithm_name', 'accuracy'
results_df = results_df.append({'data_type': 'train', 'algorithm_name': 'naive_bayes', 'accuracy': train_nb_accuracy * 100}, ignore_index=True)


# Decision Accuracy and Classification Report for Decision Tree
test_pred = gnb.predict(test_data)
test_nb_accuracy=metrics.accuracy_score(test_labels, test_pred)

print("Test Data Naive Bayes Accuracy: ", test_nb_accuracy)
print(classification_report(test_labels, test_pred))


#  'data_type', 'algorithm_name', 'accuracy'
results_df = results_df.append({'data_type': 'test', 'algorithm_name': 'naive_bayes', 'accuracy': test_nb_accuracy * 100}, ignore_index=True)


# In[19]:


# Classification Report for Hard voting based ensemble
train_pred = voting_clf_hard.predict(train_data)
train_enhard_accuracy=metrics.accuracy_score(train_labels, train_pred)

print("Train Data - Ensemble Technique and Hard Voting - Accuracy and Classification Report: ", train_enhard_accuracy)
print(classification_report(train_labels, train_pred))

#  'data_type', 'algorithm_name', 'accuracy'
results_df = results_df.append({'data_type': 'train', 'algorithm_name': 'ensemble_hard', 'accuracy': train_enhard_accuracy * 100}, ignore_index=True)


test_pred = voting_clf_hard.predict(test_data)
test_enhard_accuracy=metrics.accuracy_score(test_labels, test_pred)

print("Test Data - Ensemble Technique and Hard Voting - Accuracy and Classification Report: ", test_enhard_accuracy)
print(classification_report(test_labels, test_pred))

#  'data_type', 'algorithm_name', 'accuracy'
results_df = results_df.append({'data_type': 'test', 'algorithm_name': 'ensemble_hard', 'accuracy': test_enhard_accuracy * 100}, ignore_index=True)


# In[ ]:


# Classification Report for Soft voting based ensemble
train_pred = voting_clf_soft.predict(train_data)
train_ensoft_accuracy=metrics.accuracy_score(train_labels, train_pred)

print("Train Data - Ensemble Technique and Soft Voting - Accuracy and Classification Report: ", train_ensoft_accuracy)
print(classification_report(train_labels, train_pred))

#  'data_type', 'algorithm_name', 'accuracy'
results_df = results_df.append({'data_type': 'train', 'algorithm_name': 'ensemble_soft', 'accuracy': train_ensoft_accuracy * 100}, ignore_index=True)


test_pred = voting_clf_soft.predict(test_data)
test_ensoft_accuracy=metrics.accuracy_score(test_labels, test_pred)

print("Test Data - Ensemble Technique and Soft Voting - Accuracy and Classification Report: ", test_ensoft_accuracy)
print(classification_report(test_labels, test_pred))

#  'data_type', 'algorithm_name', 'accuracy'
results_df = results_df.append({'data_type': 'test', 'algorithm_name': 'ensemble_soft', 'accuracy': test_ensoft_accuracy * 100}, ignore_index=True)



results_df_train = results_df[(results_df["data_type"]=='train')]
print(results_df_train)



results_df_test = results_df[(results_df["data_type"]=='test')]
print(results_df_test)


# In[ ]:


# Algorithm vs Accuracies for both train and test data
sns.set(rc={'figure.figsize':(14,8)})
plt.title('Algorithms vs Accuracies', y=1.03, fontsize = 16)
ax = sns.barplot(x=results_df["algorithm_name"], y=results_df["accuracy"], hue=results_df["data_type"])
ax.set(xlabel="Different types of algorithms", ylabel = "Accuracy (in terms of percentage)")

#Matplotlib
ax=ax #annotate axis = seaborn axis
def annotateBars(row, ax=ax): 
    for p in ax.patches:
         ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
             ha='center', va='center', fontsize=13, color='blue', rotation=90, xytext=(0, 21),
             textcoords='offset points')  
plot = results_df.apply(annotateBars, ax=ax, axis=1)


# print out the most correlated feature
rndf_namescore = {}

for name, score in zip(train_data.columns, rnfc_best.feature_importances_):
    rndf_namescore[name] = score

# sorting by value in reverse order
rndf_namescore = sorted(rndf_namescore.items(), key=lambda kv: kv[1], reverse=True)

# printing it:
"Most important features that can predict Class ranked:"
rndf_namescore


# In[ ]:


# HEATMAP of test_data
plt.figure(figsize = (14,12))
#plt.title('Correlation of Numeric Features with Sale Price', y=1, size=16)
sns.heatmap(test_data.corr(), square = True, vmax=0.8)
plt.title('Test data : Heatmap of Correlation Value', y=1, size=16)


# In[ ]:


# HEATMAP of train_data
plt.figure(figsize = (14,12))
#plt.title('Correlation of Numeric Features with Sale Price', y=1, size=16)
sns.heatmap(train_data.corr(), square = True, vmax=0.8)
plt.title('train_data : Heatmap of Correlation Value', y=1, size=16)


# In[ ]:


colormap = plt.cm.viridis
plt.figure(figsize=(25,25))
plt.title('Pearson Correlation of selected Train Data Features', y=1.05, size=15)
sns.heatmap(train_data.iloc[:,:].astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
#save_fig('01_correlation')


# In[ ]:


colormap = plt.cm.viridis
plt.figure(figsize=(25,25))
plt.title('Pearson Correlation of selected Test Data Features', y=1.05, size=15)
sns.heatmap(test_data.iloc[:,:].astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
#save_fig('01_correlation')


# In[ ]:




