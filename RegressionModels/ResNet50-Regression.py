# Import necessary library to read dataset file
import numpy as np
import os
pathNow    = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
print(pathNow)

# # os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Run with CPU
# os.environ["TF_GPU_ALLOCATOR"]      = "cuda_malloc_async"
# print(os.getenv("TF_GPU_ALLOCATOR"))

unitName        = 'mmol/L'

# Create directory
smartphoneType  = 'ALL' # smartphone type each datasets (ALL, HN5T, SA31, SA72, VY12)
experimentNo    = str(1) # number of exp (0,1,2,..)
teststripType   = 'SingleTS' # urine image test strip arrangement shape (SingleTS, ALLTS, MlTTS)
reportPath      = pathNow + '/RegressionReports/ResNet50Reg_M_X_Z'
reportPath      = reportPath.replace('M', teststripType)
reportPath      = reportPath.replace('X', smartphoneType)
reportPath      = reportPath.replace('Z', experimentNo)
print(reportPath)


# Config FileName1
# ubah2 sesuai nama file hdf5
hdf5File        = pathNow + '/Datasets/Regression/Regression_SingleTS_VC_ALL.hdf5'
fileNameReport  = reportPath + '/ResNet50Output.mat' # For saving y_train, y_train_preds, y_val, y_val_preds, y_test, y_test_preds
fileNameModel   = reportPath + '/model/ResNet50.hdf5' # Save the best value model
fileNameHist    = reportPath + '/history/ResNet50.csv' # Save the history (acc, loss, val_ac, val_loss)
fileNameTable   = reportPath + '/RegressionReport.txt'
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


# Set batch size, epoch, and train-val-test ratio
batch_size     = 128
numEpochs      = 100
trainRatio     = 0.60
valRatio       = 0.20
testRatio      = 0.20
learningRate    = 1e-04
momentumVal     = 0.9

# Import necessary library for the algorithm
from tensorflow import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, ZeroPadding2D
from keras.layers import GlobalAveragePooling2D, MaxPooling2D, Add, Flatten, Dense
from keras.layers import Reshape
from keras.models import Model, load_model
from keras.utils import plot_model
from keras.initializers import glorot_uniform

initializer =  keras.initializers.glorot_uniform(seed=0)
initializer = keras.initializers.glorot_normal()

"""
Creates Residual Network with 50 layers
"""
def ResNet50(windowSizeW, windowSizeH):
    # Define the input as a tensor with shape
    inputs = Input(shape=(windowSizeW, windowSizeH, 3))
    X = inputs

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', 
               kernel_initializer=initializer)(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D((1, 1))(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 5, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f = 5, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f = 5, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')
    
    # Stage 5
    X = convolutional_block(X, f = 5, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL
    X = GlobalAveragePooling2D()(X)
    
    # output layer
    outputs = Flatten()(X)
    outputs = Dense(units=1, activation='linear')(X)  
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='resnet50')
    config = model.get_config()
    model  = keras.Model.from_config(config)

    return model

"""
Identity Block of ResNet
"""
def identity_block(X, f, filters, stage, block):
    """
    # Arguments
      f: kernel size
    """
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1,1), padding='valid', 
                            name=conv_name_base + '2a', kernel_initializer=initializer)(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    
    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1,1), padding='same', 
                            name=conv_name_base + '2b', kernel_initializer=initializer)(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)


    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1,1), padding='valid', 
               name=conv_name_base + '2c', kernel_initializer=initializer)(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

"""
Convolutional Block of ResNet
"""
def convolutional_block(X, f, filters, stage, block, s=2):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X

    # First component of main path 
    X = Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base + '2a', 
               padding='valid', kernel_initializer=initializer)(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path
    X = Conv2D(F2, (f, f), strides=(1, 1), name=conv_name_base + '2b', 
                            padding='same', kernel_initializer=initializer)(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)


    # Third component of main path
    X = Conv2D(F3, (1, 1), strides=(1, 1), name=conv_name_base + '2c', 
               padding='valid', kernel_initializer=initializer)(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X_shortcut = Conv2D(F3, (1, 1), strides=(s,s), name=conv_name_base + '1', 
                        padding='valid', kernel_initializer=initializer)(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

# Main
import numpy as np
import tensorflow as tf
from tensorflow import keras
# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()


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

# Build model
windowSizeW = feature.shape[1]
windowSizeH = feature.shape[2]
print("Dimension of image: " + str(windowSizeW) + "x" + str(windowSizeH))
model = ResNet50(windowSizeW, windowSizeH)
model.summary()
tf.keras.utils.plot_model(model, to_file=fileModelGraph, show_shapes=True)
# plot_model(model, show_shapes=True)

# Compiling the model...
# Configure the model for training...
# We use the "sparse" version of categorical_crossentropy
# because our target data is integers.

# Compile the model
# opt = keras.optimizers.SGD(learning_rate=learningRate, momentum=momentumVal)
# opt = keras.optimizers.RMSprop(learning_rate=1e-6)
opt = keras.optimizers.Adam(learning_rate=learningRate)
model.compile(loss='mae', optimizer=opt, metrics=[tf.keras.metrics.RootMeanSquaredError()])
checkpoint = keras.callbacks.ModelCheckpoint(fileNameModel, save_best_only=True, monitor='val_loss', verbose=1, mode='auto')
plateau = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr = 1e-7)


# TRAINING
print("Fit model on training data")
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=numEpochs,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(x_val, y_val),
    callbacks=[checkpoint, plateau], 
    verbose=1,
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
print('Resnet50 Regression Report')        
print(tabulate(table, headers, tablefmt="presto"))
print('\n')

with open(fileNameTable, 'w') as f:
    f.write('ResNet50 Regression Report\n')
    f.write('\n')
    f.write(tabulate(table, headers, tablefmt="presto"))

print('Done')