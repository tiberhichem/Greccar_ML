#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import
import numpy as np
import pydicom
import os
from glob import glob
import scipy.ndimage
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
from skimage.io import imread,imsave
import pandas as pd
import matplotlib.pyplot as plt
from radiomics import featureextractor
import keras
from xlrd import open_workbook
from numpy import asarray
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import brier_score_loss as bls
import PIL
from PIL import Image


# In[2]:


Main_path = 'C:\\Users\\hiche\\Desktop\\'
Train_path = 'C:\\Users\\hiche\\Desktop\\Training_data\\'
Baseline_path = 'C:\\Users\\hiche\\Desktop\\Training_data\\mart_it\\'
RAW_DATA ='C:\\Users\\hiche\\Desktop\\Training_data\\mart_it\\RAW_DATA\\'
SEG_DATA = 'C:\\Users\\hiche\\Desktop\\Training_data\\mart_it\\SEG_DATA\\'
os.chdir(Baseline_path)
os.getcwd()
bb_creation_path = 'C:\\Users\\hiche\\Desktop\\Training_data\\mart_it\\BB2D\\'


# In[ ]:


xls_table = open_workbook(Main_path+'Greccar_validation\\Tableau_Greccar4.xlsx')
df = pd.read_excel(xls_table)

# Bounding boxes creation

def readRAWSLICE(ipatient,s):
    PathDicomP = Baseline_path+str(ipatient)+"/Ax T2/"
    for dirName, subdirList, fileList in os.walk(PathDicomP):
        for filename in fileList:
            if 'IMG'+str(s).zfill(4)+'.dcm'in filename:
                ds = pydicom.read_file(os.path.join(dirName,filename))
                return ds
    return None
for i in range (1,99):
    s = df.iloc[i-1,3]
    o = df.iloc[i-1,2]
    x = df.iloc[i-1,4]
    y = df.iloc[i-1,5]
    s = int(s)
    o = int(o)
    bb_size = 120
    xmin = int(x - bb_size/2)
    xmax = int(x + bb_size/2)
    ymin = int(y - bb_size/2)
    ymax = int(y + bb_size/2)
    ds = readRAWSLICE(i,s)
    if ds is not None:
        ConstPixelDims = (int(ds.Rows), int(ds.Columns))
        ConstPixelSpacing = (float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]))
        if xmin > 0 and ymin > 0 :
            bb = ds.pixel_array[ymin:ymax,xmin:xmax]
        else:
            bb = ds.pixel_array[0:256, 0:256]

        imsave(bb_creation_path+'BB2D'+str(i)+"_"+str(o)+'.tiff',pixels)
    else :
        print ('missing_images',i)
        


# In[ ]:


# mask creation
for i in range(1,99): 
    s = df.iloc[i-1,3]
    o = df.iloc[i-1,2]
    s = int(s)
    o = int(o)
    im = sitk.ReadImage(bb_creation_path+"BB2D"+str(i)+"_"+str(o)+".tiff")
    print(im)
    ma_arr = np.ones(im.GetSize()[::-1], dtype='int')
    ma_arr[0,0]=0
    ma = sitk.GetImageFromArray(ma_arr)
    ma.CopyInformation(im)
    print(sitk.GetArrayFromImage(ma))
    sitk.WriteImage(ma, bb_creation_path+"MA"+str(i)+"_"+str(o)+".nrrd", True)


# In[ ]:


# features extraction
iteration = 0
for i in range(1,99): 
    iteration+=1
    s = df.iloc[i-1,3]
    o = df.iloc[i-1,2]
    s = int(s)
    o = int(o)
    patient = i
    imagePath = sitk.ReadImage(bb_creation_path+'BB2D'+str(i)+"_"+str(o)+'.tiff')
    labelPath = sitk.ReadImage(bb_creation_path+"MA"+str(i)+"_"+str(o)+".nrrd")
    im = sitk.GetArrayFromImage(labelPath)
    labels = np.unique(im)
    params = {'normalize': True,
            'normalizeScale': 100,
            'voxelArrayShift': 300,
            'resegmentRange': [-300, 300]}     
    extractor = featureextractor.RadiomicsFeatureExtractor(**params)
    extractor.enableImageTypeByName("Square")
    extractor.enableImageTypeByName("Wavelet")
    extractor.enableImageTypeByName("SquareRoot")
    extractor.enableImageTypeByName('LoG')
    extractor.enableImageTypeByName('Logarithm')
    extractor.enableImageTypeByName('Exponential')
    extractor.enableImageTypeByName('Gradient')
    extractor.enableImageTypeByName('LBP2D')

    result = extractor.execute(imagePath, labelPath,label = 1,label_channel = 0)
  
    if iteration == 1:
        table_result = pd.DataFrame(six.iteritems(result))
        table_result.rename(columns={0:'parameters',1:'patient_'+str(i)}, inplace=True)
    else :
        table = pd.DataFrame(six.iteritems(result))
        title = 'patient_'+str(i)
        table_result[title]=table.iloc[:,1]
    
table_fin = table_result.T


# In[ ]:


X = table_fin.iloc[1:,]
X.drop([0,1,2,3,4,5,6,7,8,9,10,14,15,16,17,20,21,22,23,24,27,28], axis=1, inplace=True)


# In[ ]:


#label creation
Y=[]
for i in range (1,99):
    o = df.iloc[i-1,2]
    o = int(o)
    Y.append(o)


# In[ ]:


# eliminate redundant feature
constant_filter = VarianceThreshold(threshold=0.15)
constant_filter.fit(X)
len(X.columns[constant_filter.get_support()])

constant_columns = [column for column in X.columns
                    if column not in X.columns[constant_filter.get_support()]]

X.drop(labels=constant_columns, axis=1, inplace=True)

# feature selection
model = ExtraTreesClassifier()
model.fit(X, Y)
rfe = RFE(model, 9)
fit = rfe.fit(X, Y)
print("Num Features: %d" % fit.n_features_)
print("Feature Ranking: %s" % fit.ranking_)
columns = X.columns[rfe.support_]
X=X.loc[:, X.columns.isin(list(columns))]
X.to_excel(Main_path+"X_train_model2.xlsx") 


# In[ ]:


# smote to resolve balance issue before training the model
sm = SMOTE(random_state=42)
X, Y = sm.fit_sample(X, Y)


# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=5)
classifier1 = RandomForestClassifier(n_estimators = 1000, random_state = 42)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X, Y):
    probas_1 = classifier1.fit(X.iloc[train], Y.iloc[train]).predict(X.iloc[test])
    probas_1 = probas_1.round()
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(Y.iloc[test], probas_1)
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


# hyperparameters tuning
gs = GridSearchCV(
estimator=RandomForestClassisifer(),
param_grid={
'max_depth': range(2,20), 
'max_features':np.arange(1,10),'n_estimators':[10, 15, 20, 50, 100, 250, 500, 1000, 5000],
'min_samples_split' : [1,3,5,8,10,12]
},
cv=5, scoring='neg_mean_squared_error', verbose=0,n_jobs=-1)
grid_result = gs.fit(X_train, Y_train)
best_params = grid_result.best_params_
random_forest = RandomForestClassifier(min_samples_split = best_params["min_samples_split"],max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"],max_features =best_params["max_features"], random_state=False, verbose=False)
random_forest.fit(X_train, Y_train)

y_pred_test = random_forest.predict(X_test)
y_pred_train = random_forest.predict(X_train)
print("Mean Squared error on training data --> {}\nMean Squared error on test data --> {}".format(bls(Y_train, y_pred_train), bls(Y_test, y_pred_test)))


# In[ ]:


# model assessment after tuning hyperparameters
rfc_predict = random_forest.predict(X_test)
rfc_predict = rfc_predict.round()
print("=== Confusion Matrix ===")
print(confusion_matrix(Y_test, rfc_predict))
print('\n')
print("=== Classification Report ===")
print(classification_report(Y_test, rfc_predict))
print('\n')
print("=== All AUC Scores ===")
print(rfc_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())

