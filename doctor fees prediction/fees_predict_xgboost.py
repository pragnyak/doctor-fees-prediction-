# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 11:55:13 2019

@author: prasad
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
reload(sys)
sys.setdefaultencoding('utf8')
dataset=pd.read_excel('Final_Train_doctor.xlsx')
dataset1=pd.read_excel('Final_Test_doctor.xlsx')

#find unique qualification in test data
qual=list(dataset1["Qualification"])
for i in range(len(qual)):
    qual[i]=str(qual[i]).replace(", ",",").strip().replace(" ","").lower()
dataset1["Qualification"]=qual
dataset1["Qualification"]=dataset1["Qualification"].str.split(",")

all_Qual_test={}
for x in dataset1["Qualification"].values:
    for each in x:
        each = each.strip()
        if each in all_Qual_test:
            all_Qual_test[each]+=1
        else:
            all_Qual_test[each]=1
all_qua_test = sorted(all_Qual_test.items(),key=lambda x:x[1],reverse=True)
final_qua =[]
for tup in all_qua_test:
    final_qua.append(str(tup[0]).strip())
    
#qualification in training data set
qual1=list(dataset["Qualification"])
for i in range(len(qual1)):
    qual1[i]=str(qual1[i]).replace(", ",",").strip().replace(" ","").lower()
dataset["Qualification"]=qual1
dataset["Qualification"]=dataset["Qualification"].str.split(",")
'''all_Qual ={}
for x in dataset["Qualification"].values:
    for each in x:
        each = each.strip()
        if each in all_Qual:
            all_Qual[each]+=1
        else:
            all_Qual[each]=1
all_qua_train = sorted(all_Qual.items(),key=lambda x:x[1],reverse=True)'''

#we are going to encode the quallifications which are present in the test dataset
#max no of qualifcations in test data is 17 

maxq=0
for k in range(len(qual)):
    if len(qual[k].split(","))>maxq:
        maxq=len(qual[k].split(","))
       
#max no on qualification in train data is 10
maxq=0
for k in range(len(qual1)):
    if len(qual1[k].split(","))>maxq:
        maxq=len(qual1[k].split(","))
        
#make ten and 17 new columns in training data set and test data and fill the qualifications 
def split_qua(ds,col,col_num):
    return ds[col].str[col_num]

for x in range(10):
    qual="qual"+str(x+1) 
    dataset[qual]=split_qua(dataset,"Qualification",x)

for x in range(17):
    qual="qual"+str(x+1) 
    dataset1[qual]=split_qua(dataset1,"Qualification",x)

#encoding qualifications
from sklearn.preprocessing import LabelEncoder 
le=LabelEncoder()
final_qua.append('nan')
le=le.fit(final_qua)
#remove all the qualifications in the train data set and not in test data
k=0
for x in range(10):
    qual="qual"+str(x+1)
    for i in range(len(dataset)):
        if dataset.at[i,qual] not in final_qua:
           dataset.at[i,qual]='nan'
           k=k+1
#label encoding on train data set   
for x in range(10):
    qual="qual"+str(x+1)
    dataset[qual]=le.transform(dataset[qual])
#replace all nan values with -1
p=0
for x in range(10):
    qual="qual"+str(x+1)
    for i in range(len(dataset)):
        if dataset.at[i,qual]==369:
           dataset.at[i,qual]=-1
           p=p+1
#replacing all nan values with 'nan' string 
s=0
for x in range(17):
    qual="qual"+str(x+1)
    for i in range(len(dataset1)):
        if dataset1.at[i,qual] not in final_qua:
           dataset1.at[i,qual]='nan'
           s=s+1
#label encoding on test data
for x in range(17):
    qual="qual"+str(x+1)
    dataset1[qual]=le.transform(dataset1[qual])
#replace all null values with -1
p=0
for x in range(17):
    qual="qual"+str(x+1)
    for i in range(len(dataset1)):
        if dataset1.at[i,qual]==369:
           dataset1.at[i,qual]=-1
           p=p+1


#extracting no of feedback from miscellaneous column  train
feedback_no=[]
import re
misc=list(dataset["Miscellaneous_Info"])
for i in range(len(misc)):
    try:
        found = re.search('%(.+?)Feedback', misc[i]).group(1)
    except :   
        found = 0
    found=str(found).strip()
    found=int(found)
    feedback_no.append(found)
dataset['feedback_no']=feedback_no

#extracting no of feedback from miscellanous column  test
feedback_no=[]
import re
misc=list(dataset1["Miscellaneous_Info"])
for i in range(len(misc)):
    try:
        found = re.search('%(.+?)Feedback', misc[i]).group(1)
    except :   
        found = 0
    found=str(found).strip()
    found=int(found)
    feedback_no.append(found)
dataset1['feedback_no']=feedback_no
    
#extracting the fees given in misc column

text=misc[62].replace(u"\u20B9","INR")
dataset["Miscellaneous_Info"]=dataset["Miscellaneous_Info"].str.replace(u"\u20B9","INR")
dataset1["Miscellaneous_Info"]=dataset1["Miscellaneous_Info"].str.replace(u"\u20B9","INR")
#training
misc_fees=[]

misc=list(dataset["Miscellaneous_Info"])
for i in range(len(misc)):
    try:
        found = re.search(r'INR(\d*)', misc[i]).group(1)
    except :   
        found = 0
    found=str(found).strip().replace(",","")
    found=int(found)
    misc_fees.append(found)
dataset['misc_fees']=misc_fees

#testing
misc_fees=[]

misc=list(dataset1["Miscellaneous_Info"])
for i in range(len(misc)):
    try:
        found = re.search(r'INR(\d*)', misc[i]).group(1)
    except :   
        found = 0
    found=str(found).strip().replace(",","")
    found=int(found)
    misc_fees.append(found)
dataset1['misc_fees']=misc_fees

           
         
#experience:training
yoe=list(dataset['Experience'])
s=0
for k in range(len(yoe)):
        yoe[k]=int(yoe[k].split(" ")[0].strip())
        s=s+1
dataset['Experience']=yoe
#rating-training
ratings=list(dataset['Rating'])
s=0
for k in range(len(ratings)):
    try:
        ratings[k]=int(ratings[k].split("%")[0].strip())
    except:
        ratings[k]=0
        s=s+1
dataset['Rating']=ratings

#experience:testing
yoe=list(dataset1['Experience'])
s=0
for k in range(len(yoe)):
        yoe[k]=int(yoe[k].split(" ")[0].strip())
        s=s+1
dataset1['Experience']=yoe
#rating-testing
ratings=list(dataset1['Rating'])
s=0
for k in range(len(ratings)):
    try:
        ratings[k]=int(ratings[k].split("%")[0].strip())
    except:
        ratings[k]=0
        s=s+1
dataset1['Rating']=ratings

#city training
dataset['Place'].fillna("none,none",inplace=True)
dataset['Place']=dataset['Place'].str.split(",")
dataset['City']=dataset['Place'].str[-1]
dataset['place']=dataset['Place'].str[0]

print(dataset[dataset["City"]=="e"].index.values)
pd.Index(dataset['City']).value_counts()
dataset["City"][3980]="none"

#city testing
dataset1['Place'].fillna("none,none",inplace=True)
dataset1['Place']=dataset1['Place'].str.split(",")
dataset1['City']=dataset1['Place'].str[-1]
dataset1['place']=dataset1['Place'].str[0]

pd.Index(dataset1['City']).value_counts()
#encoding city and profile training
dataset = pd.get_dummies(dataset,columns=["City","Profile"],prefix=["City","Profile"],drop_first=True)
#encoding city and profile training
dataset1 = pd.get_dummies(dataset1,columns=["City","Profile"],prefix=["City","Profile"])
#making a copy of both datasets
df=dataset.copy(deep=True)
df1=dataset1.copy(deep=True)

#removing unwanted columns training 
y_train=df.iloc[:,5].values
df=df.drop(["Qualification","Place","Miscellaneous_Info","place","Fees"],axis=1)
X_train=df.iloc[:,:].values

#removing unwanted columns in testing
cols = df.columns.tolist()
df1=df1[cols]
X_test=df1.iloc[:,:].values

#converting X_train and y_train as csv files 
X_tr=pd.DataFrame(X_train)
X_tr.to_csv("X_tr1.csv")
X_te=pd.DataFrame(X_test)
X_te.to_csv("X_te1.csv")
y_tr=pd.DataFrame(y_train)
y_tr.to_csv("y_tr1.csv")


#applying the model  xgboost
import xgboost as xgb
xg_reg=xgb.XGBRegressor(learning_rate =0.1,
    n_estimators=100,
    gamma=0.1,
    njobs=-1)
xg_reg.fit(X_train,y_train)
#k fold cross evaluation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = xg_reg, X = X_train, y = y_train, cv = 10)
print(accuracies.mean())
#prediction
y_pred_xgboost=xg_reg.predict(X_test)
#converting prediction to csv
ypredxgb=pd.DataFrame(y_pred_xgboost)
ypredxgb.to_csv("predictionxgboost.csv")
#applying grid search 
'''
from sklearn.model_selection import GridSearchCV
parameters = [{'learning_rate': [1,0.1,0.2,0.5], 'n_estimators': [100,1000,1500],
               'gamma': [0.1, 0.2,0.4,0]}]
grid_search = GridSearchCV(estimator = xg_reg,
                           param_grid = parameters,
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_'''
