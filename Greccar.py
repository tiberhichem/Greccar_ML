#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import
import numpy as np
import pydicom
import os
import scipy.ndimage
import scipy.fft
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
from skimage.io import imread,imsave
import numpy.ma as ma
import SimpleITK as sitk
import six
import pandas as pd
import matplotlib.pyplot as plt
import keras
from xlrd import open_workbook
from numpy import asarray
from scipy import fftpack
import pywt
import collections
from itertools import chain
import json
import logging
from radiomics import generalinfo, getFeatureClasses, getImageTypes, getParameterValidationFiles, imageoperations
from radiomics import featureextractor
from sklearn.metrics import brier_score_loss as bsl
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import ranksums
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from scipy import interp


# In[2]:


# loading the first dataset, the dataset that would serve for cross validation
Main_path = 'C:\\Users\\hiche\\Desktop\\'
Train_path = 'C:\\Users\\hiche\\Desktop\\Training_data\\'
Baseline_path = 'C:\\Users\\hiche\\Desktop\\Training_data\\mart_it\\'
RAW_DATA ='C:\\Users\\hiche\\Desktop\\Training_data\\mart_it\\RAW_DATA\\'
SEG_DATA = 'C:\\Users\\hiche\\Desktop\\Training_data\\mart_it\\SEG_DATA\\'
os.chdir(Baseline_path)
xls_table = open_workbook(Main_path+'Greccar_validation\\Tableau_Greccar4.xlsx')
df = pd.read_excel(xls_table)


# In[3]:


# features extraction for the first dataset 
iteration = 0
for i in range(1,99): 
    iteration+=1
    s = df.iloc [i-1,3] 
    o = df.iloc [i-1,2]
    s = int(s)
    o = int(o)
    patient = i
    imagePath = sitk.ReadImage(RAW_DATA+"RAW_"+str(i)+"_"+str(o)+".tiff")
    labelPath = sitk.ReadImage(SEG_DATA+"SEG_"+str(i)+"_"+str(o)+".tiff")
    params = {
 
            'normalize': True,
            'normalizeScale': 100,
            'voxelArrayShift': 300,
            'resegmentRange': [-300, 300],
           
    }     
    extractor = featureextractor.RadiomicsFeatureExtractor(**params)
    extractor.enableImageTypeByName("Square")
    extractor.enableImageTypeByName("Wavelet")
    extractor.enableImageTypeByName("SquareRoot")
    extractor.enableImageTypeByName('LoG')
    extractor.enableImageTypeByName('Logarithm')
    extractor.enableImageTypeByName('Exponential')
    extractor.enableImageTypeByName('Gradient')
    extractor.enableImageTypeByName('LBP2D')
    result = extractor.execute(imagePath, labelPath)
    if iteration == 1:
        table_result = pd.DataFrame(six.iteritems(result))
        table_result.rename(columns={0:'parameters',1:'patient_'+str(i)}, inplace=True)
    else :
        table = pd.DataFrame(six.iteritems(result))
        title = 'patient_'+str(i)
        table_result[title]=table.iloc[:,1]
    
    table_fin = table_result.T

Field_Strength_1 = df.iloc[:98,34] # load field strength
table_fin_1 = table_fin.iloc[1:,:]


# In[4]:


# loading the second dataset

Main_path = 'C:\\Users\\hiche\\Desktop\\'
Baseline_path = 'C:\\Users\\hiche\\Desktop\\Baseline\\'
RAW_DATA ='C:\\Users\\hiche\\Desktop\\Baseline\\RAW_DATA\\'
SEG_DATA = 'C:\\Users\\hiche\\Desktop\\Baseline\\SEG_DATA\\'
os.chdir(Baseline_path)

xls_table = open_workbook(Main_path+'Greccar_validation\\Tableau_test.xlsx')
df = pd.read_excel(xls_table)


# In[5]:


# feature extraction for the second dataset

iteration = 0
for i in range(1,49): 
    print (i)
    iteration+=1
    s = df.iloc [i-1,3] 
    o = df.iloc [i-1,2] 
    s = int(s+1)
    o = int(o)
    patient = i
    imagePath = sitk.ReadImage(RAW_DATA+"RAW_"+str(i)+"_"+str(o)+".tiff")
    labelPath = sitk.ReadImage(SEG_DATA+"SEG_"+str(i)+"_"+str(o)+".tiff")
    params = {'normalize': True,
            'normalizeScale': 100,
            'voxelArrayShift': 300,
            'resegmentRange': [-300, 300],

             }     
    extractor = featureextractor.RadiomicsFeatureExtractor(**params)
    extractor.enableImageTypeByName("Square")
    extractor.enableImageTypeByName("Wavelet")
    extractor.enableImageTypeByName("SquareRoot")
    extractor.enableImageTypeByName('LoG')
    extractor.enableImageTypeByName('Logarithm')
    extractor.enableImageTypeByName('Exponential')
    extractor.enableImageTypeByName('Gradient')
    extractor.enableImageTypeByName('LBP2D')
    result = extractor.execute(imagePath, labelPath)
    if iteration == 1:
        table_result = pd.DataFrame(six.iteritems(result))
        table_result.rename(columns={0:'parameters',1:'patient_'+str(i)}, inplace=True)
    else :
        table = pd.DataFrame(six.iteritems(result))
        title = 'patient_'+str(i)
        table_result[title]=table.iloc[:,1]

    table_fin_2 = table_result.T
    
Field_Strength_2 = df.iloc[:49,13] # load field strength
table_fin_2 = table_fin_2.iloc[1:,:]


# In[6]:


table_fin_1.drop([0,1,2,3,4,5,6,7,8,9,10,14,15,16,17,20,21,22,23,24,27,28],axis=1, inplace=True)
table_fin_2.drop([0,1,2,3,4,5,6,7,8,9,10,14,15,16,17,20,21,22,23,24,27,28],axis=1, inplace=True)


# In[7]:


# removal of redundant features (with low variance)
constant_filter = VarianceThreshold(threshold=0.15)
constant_filter.fit(table_fin_1)
len(table_fin_1.columns[constant_filter.get_support()])

constant_columns = [column for column in table_fin_1.columns
                    if column not in table_fin_1.columns[constant_filter.get_support()]]

table_fin_1.drop(labels=constant_columns, axis=1, inplace=True)


# In[15]:


# loading outcomes
xls_table = open_workbook(Main_path+'Radiomics_signature\\Outcomes.xlsx')
df = pd.read_excel(xls_table)
Y=[]
for i in range (1,147):
    o = df.iloc[i-1,1]
    o = int(o)
    Y.append(o)
Y1 = Y[:98]
Y2 = Y[98:]


# In[9]:


# feature selection
model = ExtraTreesClassifier()
model.fit(table_fin_1, Y1)
rfe = RFE(model, 9) 
fit = rfe.fit(table_fin_1, Y1)
columns = table_fin_1.columns[rfe.support_]
table_fin_1 = table_fin_1.loc[:, table_fin_1.columns.isin(list(columns))]
table_fin_2 = table_fin_2.loc[:, table_fin_2.columns.isin(list(columns))]


# In[14]:


len(y_test)


# In[17]:


# setting the GridSearch for hyperparameters tuning 
gs = GridSearchCV(
estimator=RandomForestClassifier(),
param_grid={
'max_depth': range(2,10), 
'max_features':np.arange(1,10),'n_estimators':[100,500,1000],
},
cv=5, scoring='brier_score_loss', verbose=0,n_jobs=-1)

X_train = table_fin_1
X_test = table_fin_2
y_train = Y1
y_test = Y2

sm = SMOTE(random_state=42) # smote to solve the imbalance issue
X_train_sm, y_train_sm = sm.fit_sample(X_train, y_train)
grid_result = gs.fit(X_train_sm, y_train_sm)
best_params = grid_result.best_params_
random_forest = RandomForestClassifier(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"],max_features =best_params["max_features"], random_state=False, verbose=False)
random_forest.fit(X_train_sm, y_train_sm)
y_pred_train_sm = random_forest.predict(X_train_sm)
random_forest.fit(X_train_sm, y_train_sm)
X_test_sm, y_test_sm = sm.fit_sample(X_test, y_test)
y_pred_sm = random_forest.predict(X_test_sm)

# model performance metrics
cm1 = confusion_matrix(y_train_sm, y_pred_train_sm)
cm2 = confusion_matrix(y_test_sm, y_pred_sm)
print(accuracy_score(y_test_sm, y_pred_sm), accuracy_score(y_train_sm, y_pred_train_sm))
print(cm1)
print(cm2)


# In[19]:


# checking the relationship between selected features and magnetic field strength
X = pd.concat([X_train, X_test], axis = 0)
Field_Strength = pd.concat([X_train, X_test], axis = 0)
for i in range (1,10):
    print(ranksums(X.iloc[:,i-1], Field_Strength))


# In[ ]:




