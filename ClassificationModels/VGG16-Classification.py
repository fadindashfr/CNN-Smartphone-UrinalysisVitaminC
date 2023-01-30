
import os
# os.environ["TF_GPU_ALLOCATOR"]      = "cuda_malloc_async"
# print(os.getenv("TF_GPU_ALLOCATOR"))
# from cachetools import LRUCache

pathNow    = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
print(pathNow)

numClass        = 5
className       = ['0', '0.6', '1.4', '2.8', '5.6']
unitName        = 'mmol/L'

# Hyperparameters
numEpochs       = 50
batch_size      = 32
trainRatio      = 0.40
valRatio        = 0.30
testRatio       = 0.30
learningRate    = 1e-04
momentumVal     = 0.9

# Create directory
smartphoneType  = 'ALL' # smartphone type each datasets (ALL, HN5T, SA31, SA72, VY12)
experimentNo    = str(1) # number of exp (0,1,2,..)
teststripType   = 'SingleTS' # urine image test strip arrangement shape (SingleTS, ALLTS, MlTTS)
reportPath      = pathNow + '/ClassificationReports/VGG16Class_M_X_Z'
reportPath      = reportPath.replace('M', teststripType)
reportPath      = reportPath.replace('X', smartphoneType)
reportPath      = reportPath.replace('Z', experimentNo)
print(reportPath)

# Config FileName
hdf5File        = pathNow + '/Datasets/Classification/Classification_SingleTS_VC_ALL.hdf5'
fileNameReport  = reportPath + '/VGG16Output.mat' #  For saving y_train, y_train_preds, y_val, y_val_preds, y_test, y_test_preds
fileNameModel   = reportPath + '/model/VGG16.hdf5' # Save the best value model
fileNameHist    = reportPath + '/history/VGG16History.csv' # Save the history (acc, loss, val_ac, val_loss)
fileModelGraph  = reportPath + '/model/Structure.png'
fileNameTable   = reportPath + '/ClassificationReport.txt'

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
# from keras.utils import plot_model
from keras.layers import Conv2D, BatchNormalization, Flatten, Dense, MaxPooling2D, Dropout

def getVGG16Class(windowSizeW, windowSizeH, numClass):
    inputs = Input(shape=(windowSizeW, windowSizeH, 3))
    #size = (windowSizeW, windowSizeH) # Minimum size = 32x32 (include top & non include top)
    x = Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu')(inputs)
    x = Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(1,1), strides=(1,1))(x)

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
    x = Dense(units=512, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    outputs = Dense(units=numClass, activation='softmax')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs, name='VGG16')
    config = model.get_config() # To save custom objects to HDF5
    model = keras.Model.from_config(config)
    return model

# -----------------------------------------------------------------------
# Main

import numpy as np
import tensorflow as tf
from myUtils import loadData
feature, target = loadData(hdf5File)
target = np.ravel(target)

# change target: 0 to class-1
target = target - 1
target = tf.keras.utils.to_categorical(target,  num_classes=numClass)

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
model = getVGG16Class(windowSizeW, windowSizeH, numClass)
model.summary()
tf.keras.utils.plot_model(model, to_file=fileModelGraph, show_shapes=True)

opt = keras.optimizers.Adam(learning_rate=learningRate, beta_1 = 0.9, beta_2 = 0.99, epsilon = 1e-08)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
checkpoint = keras.callbacks.ModelCheckpoint(fileNameModel, save_best_only=True, monitor='val_loss', verbose=1, mode='auto')
plateau = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr = 1e-7)

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

# Plot learning curve
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

plt.figure()
plt.plot(history.history['accuracy'], label='Training', color='blue', linestyle='-', linewidth=1)
plt.plot(history.history['val_accuracy'], label='Validation', color='red', linestyle=':', linewidth=1)
plt.title("Accuration", loc='center')
plt.xlabel("Epochs")
plt.xticks(np.arange(0,numEpochs+numEpochs/10,numEpochs/10))
plt.ylabel("Accuration")
plt.yticks(np.arange(0,1.1,0.1))
plt.axis([0,numEpochs,0,1])
plt.legend(loc="lower right")
plt.savefig(reportPath + '/graph/1. Akurasi.png')

plt.figure()
plt.plot(history.history['loss'], label='Training', color='blue', linestyle='-', linewidth=1)
plt.plot(history.history['val_loss'], label='Validation', color='red', linestyle=':', linewidth=1)
plt.title("Loss", loc='center')
plt.xlabel("Epochs")
plt.xticks(np.arange(0,numEpochs+numEpochs/10,numEpochs/10))
plt.ylabel("Loss")
tlval = max(history.history['loss'])
vlval = max(history.history['val_loss'])
maxvalue = max(tlval, vlval)
plt.yticks = (np.arange(0,maxvalue+maxvalue/10, maxvalue/10))
plt.legend(loc="upper right")
plt.axis([0,numEpochs,0,maxvalue])
plt.savefig(reportPath + '/graph//2. Loss.png')

# Evaluate using training data
y_train_preds = model.predict(x_train)
y_train_preds = np.argmax(y_train_preds, axis=1) # axis = 1: mencari indeks yg nilainya maksimum per baris

# Evaluate using validation data
y_val_preds = model.predict(x_val)
y_val_preds = np.argmax(y_val_preds, axis=1)

# Evaluate using testing data
y_test_preds = model.predict(x_test)
y_test_preds = np.argmax(y_test_preds, axis=1)

y_train = (np.argmax(y_train, axis=1))
y_val = (np.argmax(y_val, axis=1))
y_test = (np.argmax(y_test, axis=1))
# Save to integer format
y_train.astype(int)
y_train_preds.astype(int)
y_val.astype(int)
y_val_preds.astype(int)
y_test.astype(int)
y_test_preds.astype(int)

# Plot Confusion Matrix
y_act = y_test
y_pred = y_test_preds

cm = confusion_matrix(y_act, y_pred)
cm_df = pd.DataFrame(cm, index = className, columns = className)

plt.figure()
sb.heatmap(cm_df, annot=True, fmt="d")
plt.title('Confusion Matrix')
plt.ylabel('Actual (' + unitName + ')')
plt.xlabel('Predicted (' + unitName + ')')
# plt.xticks(rotation = 68)
plt.savefig(reportPath + '/graph/3. Confusion Matrix.png', dpi=300)

accTrain = tf.keras.metrics.Accuracy()
accTrain.update_state(y_train, y_train_preds)
accVal = tf.keras.metrics.Accuracy()
accVal.update_state(y_val, y_val_preds)
accTest = tf.keras.metrics.Accuracy()
accTest.update_state(y_test, y_test_preds)

# Create Classification Report
from tabulate import tabulate

headers = ["Metrics", "Training", "Validation", "Testing"]
table = [
        ["Accuracy",accTrain.result().numpy(),accVal.result().numpy(),accTest.result().numpy()]
        ]
print('\n')
print('VGG16 Classification Report')        
print(tabulate(table, headers, tablefmt="presto"))
print('\n')

with open(fileNameTable, 'w') as f:
    f.write('Classification VGG16 Report\n')
    f.write('\n')
    f.write(tabulate(table, headers, tablefmt="presto"))

# Saving data to file
from scipy.io import savemat

matData = {"YTr": y_train, "YTrP": y_train_preds, "YV": y_val, "YVP": y_val_preds, "YTt": y_test, "YTtP": y_test_preds}
savemat(fileNameReport, matData)

# Convert the history.history dict to a pandas dataframe
hist_df = pd.DataFrame(history.history)

# Save history to csv
with open(fileNameHist, mode='w') as f:
    hist_df.to_csv(f)

print('\nCLASSIFICATION REPORTS\n')
print(classification_report(y_act, y_pred, target_names=className))

print('Done')