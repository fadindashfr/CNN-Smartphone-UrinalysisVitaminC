import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Run with CPU
# os.environ["TF_GPU_ALLOCATOR"]      = "cuda_malloc_async"
# print(os.getenv("TF_GPU_ALLOCATOR"))

pathNow    = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
print(pathNow)

unitName        = 'mmol/L'

# Hyperparameters
numEpochs       = 100
batch_size      = 128
trainRatio      = 0.60
valRatio        = 0.20
testRatio       = 0.20
learningRate    = 1e-4
momentumVal     = 0.9

# Create directory
smartphoneType  = 'ALL'  # smartphone type each datasets (ALL, HN5T, SA31, SA72, VY12)
experimentNo    = str(1)  # number of exp (0,1,2,..)
teststripType   = 'SingleTS' # urine image test strip arrangement shape (SingleTS, ALLTS, MlTTS)
reportPath      = pathNow + '/RegressionReports/VGG16Reg_M_X_Z'
reportPath      = reportPath.replace('M', teststripType)
reportPath      = reportPath.replace('X', smartphoneType)
reportPath      = reportPath.replace('Z', experimentNo)
print(reportPath)

# Config FileName
hdf5File    = pathNow + '/Datasets/Regression/Regression_SingleTS_VC_ALL.hdf5' #dataset
fileNameReport  = reportPath + '/VGG16Output.mat' # For saving y_train, y_train_preds, y_val, y_val_preds, y_test, y_test_preds
fileNameTable   = reportPath + '/RegressionReport.txt'
fileNameModel   = reportPath + '/model/VGG16.hdf5' # Save the best value model
fileNameHist    = reportPath + '/history/History.csv' # Save the history (acc, loss, val_ac, val_loss)
fileModelGraph  = reportPath + '/model/Structure.png'
fileRMSEGraph   = reportPath + '/graph/1. RMSE.png'
fileLossGraph   = reportPath + '/graph/2. Loss.png'
fileCCGraph     = reportPath + '/graph/3. CorrelationCurve.png'

import shutil
if os.path.exists(reportPath):
    print('There is directory with the same name before. So it will be removed')
    shutil.rmtree(reportPath, ignore_errors=False, onerror=None)
    print('Succesfully removed directory')

try:
    print('Create new directory')
    os.makedirs(reportPath + '/graph')
    os.makedirs(reportPath + '/history')
    os.makedirs(reportPath + '/model')
except OSError:
    print('Creation of new directory failed')
else:
    print('Successfully created new directory')

# Prepare CNN Model: VGG16
import tensorflow as tf
from tensorflow import keras
from keras import Model, Input, layers
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
from keras.layers import BatchNormalization, Dropout

def getVGG16Regression(windowSizeW, windowSizeH):
    inputs = Input(shape=(windowSizeW, windowSizeH, 3))
    #size = (windowSizeW, windowSizeH) # Minimum size = 32x32 (include top & non include top)
    x = Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu')(inputs)
    x = Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(1,1))(x)

    x = Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)

    x = Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu')(x)  
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)

    x = Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu')(x)  
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
    
    x = Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)

    # If include top
    x = Flatten()(x)
    x = Dense(units=4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(units=4096, activation='relu')(x)
    x = Dropout(0.5)(x)

    outputs = Dense(units=1, activation='linear')(x)  

    # Define the model
    model = Model(inputs=inputs, outputs=outputs, name='VGG16')
    config = model.get_config() # To save custom objects to HDF5
    model = keras.Model.from_config(config)
    return model

# -----------------------------------------------------------------------
# Main

import numpy as np
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from myUtils import loadData

feature, target = loadData(hdf5File)
target = np.ravel(target)

print("Number of samples:" + str(feature.shape[0]))
print("Feature Data Dimension :" + str(feature.shape) + ", Target :" + str(target.shape))

from myUtils import splitTrainValTest

x_train, x_val, x_test, y_train, y_val, y_test = splitTrainValTest(feature, target, trainRatio=trainRatio, valRatio=valRatio, testRatio=testRatio)
print("Num of training data: " + str(x_train.shape))
print("Num of val data: " + str(x_val.shape))
print("Num of test data: " + str(x_test.shape))

# Build the model
windowSizeW = feature.shape[1]
windowSizeH = feature.shape[2]
print("Dimension of image: " + str(windowSizeW) + "x" + str(windowSizeH))
model = getVGG16Regression(windowSizeW, windowSizeH)
model.summary()
tf.keras.utils.plot_model(model, to_file=fileModelGraph, show_shapes=True)

# Compile the model
opt = keras.optimizers.Adam(
    learning_rate=learningRate)

model.compile(
    loss='mae', 
    optimizer=opt, 
    metrics=[tf.keras.metrics.RootMeanSquaredError()])

checkpoint = keras.callbacks.ModelCheckpoint(
    fileNameModel, 
    save_best_only=True, 
    monitor='val_loss', 
    verbose=1, 
    mode='auto')

plateau = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=5, 
    min_lr = 1e-7)

# Training
print("Fit model on training data - VGG16")
history = model.fit(
    x_train,
    y_train,
    batch_size = batch_size,
    epochs = numEpochs,
    validation_data = (x_val, y_val),
    callbacks = [checkpoint, plateau],
    verbose = 1,
    shuffle=True
)

# Evaluate using training data
y_train_preds   = model.predict(x_train)

# Evaluate using validation data
y_val_preds     = model.predict(x_val)

# Evaluate using testing data
y_test_preds    = model.predict(x_test)

# # Evaluate using urine sample
# y_urine_preds = model.predict(feature_val)

# Plot learning curve
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import math
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

plt.figure()
plt.plot(history.history['root_mean_squared_error'], label='Training', color='blue', linestyle='-', linewidth=1)
plt.plot(history.history['val_root_mean_squared_error'], label='Validation', color='red', linestyle=':', linewidth=1)
plt.title("Root Mean Squared Error", loc='center')
plt.xlabel("RMSE")
plt.xticks(np.arange(0,numEpochs+numEpochs/10,numEpochs/10))
plt.ylabel("RMSE")
rmseTrain = max(history.history['root_mean_squared_error'])
rmseVal = max(history.history['val_root_mean_squared_error'])
maxvalue = max(rmseTrain, rmseVal)
plt.yticks = (np.arange(0,maxvalue+maxvalue/10, maxvalue/10))
plt.axis([0,numEpochs,0,maxvalue+maxvalue/10])
plt.legend(loc="lower right")
plt.savefig(fileRMSEGraph)

plt.figure()
plt.plot(history.history['loss'], label='Training', color='blue', linestyle='-', linewidth=1)
plt.plot(history.history['val_loss'], label='Validation', color='red', linestyle=':', linewidth=1)
plt.title("Loss", loc='center')
plt.xlabel("Epochs")
plt.xticks(np.arange(0,numEpochs+numEpochs/10,numEpochs/10))
plt.ylabel("Loss")
lossTrain = max(history.history['loss'])
lossVal = max(history.history['val_loss'])
maxvalue = max(lossTrain, lossVal)
plt.yticks = (np.arange(0,maxvalue+maxvalue/10, maxvalue/10))
plt.legend(loc="upper right")
plt.axis([0,numEpochs,0,maxvalue+maxvalue/10])
plt.savefig(fileLossGraph)

# Create Regression Curve
maxVal = math.ceil(max(max(y_train),max(y_train_preds)))
xNorm = np.arange(0, maxVal, maxVal/100)
yNorm = np.arange(0, maxVal, maxVal/100)

fig = plt.figure()
ax = fig.add_subplot()
plt.scatter(y_train, y_train_preds,label='Training', color='blue')
plt.scatter(y_val, y_val_preds,label='Validation', color='green')
plt.plot(xNorm, yNorm, color='red')
plt.title("Regression Curve", loc='center')
plt.xlabel("Measurement")
plt.ylabel("Predicted")
plt.axis([0,maxVal,0,maxVal])
plt.legend(loc="lower right")
ax.set_aspect('equal', adjustable='box')
plt.savefig(fileCCGraph)

# Saving data to file
from scipy.io import savemat

matData = {"YTr": y_train, "YTrP": y_train_preds, "YV": y_val, "YVP": y_val_preds, "YTt": y_test, "YTtP": y_test_preds}
savemat(fileNameReport, matData)

# Convert the history.history dict to a pandas dataframe
hist_df = pd.DataFrame(history.history)

# Save history to csv
with open(fileNameHist, mode='w') as f:
    hist_df.to_csv(f)

rmseTrain   = tf.keras.metrics.RootMeanSquaredError()
rmseTrain.update_state(y_train,y_train_preds)
rmseVal     = tf.keras.metrics.RootMeanSquaredError()
rmseVal.update_state(y_val,y_val_preds)
rmseTest    = tf.keras.metrics.RootMeanSquaredError()
rmseTest.update_state(y_test,y_test_preds)

csTrain     = tf.keras.metrics.CosineSimilarity()
csTrain.update_state(y_train,y_train_preds)
csVal       = tf.keras.metrics.CosineSimilarity()
csVal.update_state(y_val,y_val_preds)
csTest      = tf.keras.metrics.CosineSimilarity()
csTest.update_state(y_test,y_test_preds)

mseTrain    = mean_squared_error(y_train, y_train_preds)
mseVal      = mean_squared_error(y_val, y_val_preds)
mseTest     = mean_squared_error(y_test, y_test_preds)

mapeTrain    = mean_absolute_percentage_error(y_train, y_train_preds)
mapeVal      = mean_absolute_percentage_error(y_val, y_val_preds)
mapeTest     = mean_absolute_percentage_error(y_test, y_test_preds)

R2Train     = r2_score(y_train,y_train_preds)
R2Val       = r2_score(y_val,y_val_preds)
R2Test      = r2_score(y_test,y_test_preds)

# Create Regression Report
from tabulate import tabulate

headers = ["Metrics", "Training", "Validation", "Testing"]
table = [
        ["Root Mean Squared Error",rmseTrain.result().numpy(), rmseVal.result().numpy(), rmseTest.result().numpy()],
        ["Cosine Similarity",csTrain.result().numpy(),csVal.result().numpy(),csTest.result().numpy()],
        ["Mean Squared Error",mseTrain,mseVal,mseTest],
        ["Mean Absolute Percentage Error",mapeTrain,mapeVal,mapeTest],
        ["Coefficient of Determination",R2Train,R2Val,R2Test]
        ]

print('\n')
print('VGG16 Regression Report')        
print(tabulate(table, headers, tablefmt="presto"))
print('\n')

with open(fileNameTable, 'w') as f:
    f.write('VGG16 Regression Report\n')
    f.write('\n')
    f.write(tabulate(table, headers, tablefmt="presto"))

print('Done')