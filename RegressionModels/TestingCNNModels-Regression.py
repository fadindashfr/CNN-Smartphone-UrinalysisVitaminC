
#%%from cv2 import transpose
import tensorflow as tf
import matplotlib.pyplot as plt
import os, math

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Run with CPU
# os.environ["TF_GPU_ALLOCATOR"]      = "cuda_malloc_async"
# print(os.getenv("TF_GPU_ALLOCATOR"))

from myUtils import loadData
from time import time
from scipy.io import loadmat
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import r2_score

pathNow    = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
print(pathNow)

file_path = pathNow + '/Datasets/Regression_Val/SingleTS.hdf5' # testing dataset with real urine
smartphoneType  = 'ALL' # ubah sesuai dataset (ALL, HN5T, SA31, SA72, VY12)
experimentNo    = str(1) # smartphone type each datasets (ALL, HN5T, SA31, SA72, VY12)
teststripType   = 'SingleTS' # urine image test strip arrangement shape (SingleTS, ALLTS, MlTTS)
cnnmodels       = 'ResNet50Reg' # ResNet50 or VGG16
reportPath      = pathNow + '/RegressionTestingReports/M/E_M_X_Z'
reportPath      = reportPath.replace('E', cnnmodels)
reportPath      = reportPath.replace('M', teststripType)
reportPath      = reportPath.replace('X', smartphoneType)
reportPath      = reportPath.replace('Z', experimentNo)
fileNameTable   = reportPath + '/RegressionReport.txt'

import shutil
if os.path.exists(reportPath):
    print('There is directory with the same name before. So it will be removed')
    shutil.rmtree(reportPath, ignore_errors=False, onerror=None)
    print('Succesfully removed directory')

try:
    print('Create new directory')
    os.makedirs(reportPath + '/graph')
except OSError:
    print('Creation of new directory failed')
else:
    print('Successfully created new directory')

# SINGLETS
model_path = pathNow + "/RegressionReports/SingleTS/ResNet50Reg_SingleTS_ALL_2/model/ResNet50.hdf5"
# model_path = pathNow + "/RegressionReports/SingleTS/VGG16Reg_SingleTS_ALL_2/model/VGG16.hdf5"

# MULTIPLE TS
# model_path = pathNow + "/RegressionReports/MultipleTS/ResNet50Reg_MLTTS_ALL_2/model/ResNet50.hdf5"
# model_path = pathNow + "/RegressionReports/MultipleTS/VGG16_Reg_MultipleTS_VC_1/model/VGG16.hdf5"

# ALL TS
# model_path = pathNow + "/RegressionReports/AllTS/ResNet50Reg_ALLTS_ALL_1/model/ResNet50.hdf5"
# model_path = pathNow + "/RegressionReports/AllTS/VGG16_Reg_ALLTS_VC_1/model/VGG16.hdf5"

imgs, u_gt = loadData(file_path)
imgs = np.transpose(imgs, axes=[3, 2, 1, 0])
print(imgs.shape, u_gt.shape)
u_gt = np.ravel(u_gt)
print(imgs.shape, u_gt.shape)


saved_model = tf.keras.models.load_model(model_path)
predicted = saved_model.predict(imgs) #predict
print(predicted.shape)

rmse_val = tf.keras.metrics.RootMeanSquaredError()
rmse_val.update_state(u_gt, predicted)
print('RMSE Urine: ' + str(rmse_val.result().numpy()))

R2_val= r2_score(u_gt, predicted)
print('R2 score Val: ', R2_val)

# Create Regression Curve
maxVal = math.ceil(max(max(u_gt),max(predicted)))
xNorm = np.arange(0, maxVal, maxVal/100)
yNorm = np.arange(0, maxVal, maxVal/100)
fig = plt.figure()
ax = fig.add_subplot()
plt.scatter(u_gt, predicted,label='validationTest', color='blue')
plt.plot(xNorm, yNorm, color='red')
plt.title("Regression Curve", loc='center')
plt.xlabel("Measured Vitamin C (mmol/L)")
plt.ylabel("Predicted Vitamin C (mmol/L)")
plt.axis([0,maxVal,0,maxVal])
plt.legend(loc="lower right")
ax.set_aspect('equal', adjustable='box')
fileCCGraph     = reportPath + '/graph/1. CorrelationCurve_1.png' #GANTIGANTI
plt.savefig(fileCCGraph)

# Create Regression Report
from tabulate import tabulate

headers = ["Metrics", "Validation"]
table = [
        ["Root Mean Squared Error",rmse_val.result().numpy()],
        ["Coefficient of Determination",R2_val]
        ]

print('\n')
print('Validation Regression Report')        
print(tabulate(table, headers, tablefmt="presto"))
print('\n')

with open(fileNameTable, 'w') as f:
    f.write('Validation Regression Report\n')
    f.write('\n')
    f.write(tabulate(table, headers, tablefmt="presto"))

print('Done')
# %%
