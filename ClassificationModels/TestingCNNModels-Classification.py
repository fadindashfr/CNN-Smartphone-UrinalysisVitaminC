
#%%from cv2 import transpose
import tensorflow as tf
import matplotlib.pyplot as plt
import os

#  os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Run with CPU
# os.environ["TF_GPU_ALLOCATOR"]      = "cuda_malloc_async"
# print(os.getenv("TF_GPU_ALLOCATOR"))

import h5py
from myUtils import loadData
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sb

pathNow    = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
print(pathNow)

file_path = pathNow + '/Datasets/Classification_Val/SingleTS.hdf5' # testing dataset with real urine
smartphoneType  = 'ALL' # smartphone type each datasets (ALL, HN5T, SA31, SA72, VY12)
experimentNo    = str(1) # number of exp (0,1,2,..)
teststripType   = 'SingleTS' # urine image test strip arrangement shape (SingleTS, ALLTS, MlTTS)
cnnmodels       = 'ResNet50Class' # ResNet50 or VGG16
reportPath      = pathNow + '/ClassificationTestingReports/M/E_M_X_Z'
reportPath      = reportPath.replace('E', cnnmodels)
reportPath      = reportPath.replace('M', teststripType)
reportPath      = reportPath.replace('X', smartphoneType)
reportPath      = reportPath.replace('Z', experimentNo)
fileNameTable   = reportPath + '/ClassificationReport.txt'

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
model_path = pathNow + "/ClassificationReports/SingleTS/ResNet50Class_SingleTS_ALL_2/model/ResNet50.hdf5"
# model_path = pathNow + "/ClassificationReports/SingleTS/VGG16Class_SingleTS_ALL_1/model/VGG16.hdf5"

# MULTIPLE TS
# model_path = pathNow + "/ClassificationReports/MultipleTS/ResNet50Class_MLTTS_ALL_1/model/ResNet50.hdf5"
# model_path = pathNow + "/ClassificationReports/MultipleTS/VGG16Class_MLTTS_ALL_1/model/VGG16.hdf5"

# ALL TS
# model_path = pathNow + "/ClassificationReports/AllTS/ResNet50Class_ALLTS_ALL_1/model/ResNet50.hdf5"
# model_path = pathNow + "/ClassificationReports/AllTS/VGG16Class_AllTS_ALL_3/model/VGG16.hdf5"

imgs, u_gt = loadData(file_path)
imgs = np.transpose(imgs, axes=[3, 2, 1, 0])
print(imgs.shape, u_gt.shape)
u_gt = np.ravel(u_gt)
print(imgs.shape, u_gt.shape)

# change target: 0 to class-1
numClass = 5
className       = ['0', '0.6', '1.4', '2.8', '5.6']
unitName        = 'mmol/L'
u_gt = u_gt - 1
# u_gt = tf.keras.utils.to_categorical(u_gt,  num_classes=numClass)

print(imgs.shape, u_gt.shape)

saved_model = tf.keras.models.load_model(model_path)
predicted = saved_model.predict(imgs) #predict
predicted = np.argmax(predicted, axis=1)
print(predicted.shape)

predicted.astype(int)

acc_val = tf.keras.metrics.Accuracy()
acc_val.update_state(u_gt, predicted)
print('Accuracy: ' + str(acc_val.result().numpy()))

# Create Classification Report
from tabulate import tabulate

headers = ["Metrics", "Validation"]
table = [
        ["Accuracy",acc_val.result().numpy()]
        ]

# Plot Confusion Matrix
y_act = u_gt
y_pred = predicted

cm = confusion_matrix(y_act, y_pred)
cm_df = pd.DataFrame(cm, index = className, columns = className)

plt.figure()
sb.heatmap(cm_df, annot=True, fmt="d",cmap="rocket")
plt.title('Confusion Matrix CNN')
plt.ylabel('Actual Vitamin C (mmol/L)')
plt.xlabel('Predicted Vitamin C (mmol/L)')
# plt.xticks(rotation = 68)
plt.savefig(reportPath + '/graph/1. Confusion Matrix.png', dpi=300)

print('\n')
print('Validation Classsification Report')        
print(tabulate(table, headers, tablefmt="presto"))
print('\n')

with open(fileNameTable, 'w') as f:
    f.write('Validation Classification Report\n')
    f.write('\n')
    f.write(tabulate(table, headers, tablefmt="presto"))

print('\nCLASSIFICATION REPORTS\n')
print(classification_report(y_act, y_pred, target_names=className, zero_division = 0))

print('Done')
# %%
