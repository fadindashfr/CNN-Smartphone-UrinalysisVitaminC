# Import necessary library to read dataset file
import numpy as np
import os
pathNow    = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
print(pathNow)

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Run with CPU
# os.environ["TF_GPU_ALLOCATOR"]      = "cuda_malloc_async"
# print(os.getenv("TF_GPU_ALLOCATOR"))

numClass      = 5
className       = ['0', '0.6', '1.4', '2.8', '5.6']
unitName        = 'mmol/L'


# Create directory
smartphoneType  = 'ALL' # smartphone type each datasets (ALL, HN5T, SA31, SA72, VY12)
experimentNo    = str(1) # number of exp (0,1,2,..)
teststripType   = 'SingleTS'  # urine image test strip arrangement shape (SingleTS, ALLTS, MlTTS)
reportPath      = pathNow + '/ClassificationReports/ResNet50Class_M_X_Z'
reportPath      = reportPath.replace('M', teststripType)
reportPath      = reportPath.replace('X', smartphoneType)
reportPath      = reportPath.replace('Z', experimentNo)
print(reportPath)


# Config FileName
# ubah2 sesuai nama file hdf5
hdf5File        = pathNow + '/Datasets/Classification/Classification_SingleTS_VC_ALL.hdf5'
fileNameReport  = reportPath + '/ResNet50Output.mat' # Untuk menyimpan y_train, y_train_preds, y_val, y_val_preds, y_test, y_test_preds
fileNameModel   = reportPath + '/model/ResNet50.hdf5' # nyimpen best value model
fileNameHist    = reportPath + '/history/ResNet50.csv' # Untuk menyimpan history (acc, loss, val_ac, val_loss)
fileNameTable   = reportPath + '/ClassificationReport.txt'
fileModelGraph  = reportPath + '/model/Structure.png'

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
batch_size     = 32
numEpochs      = 50
trainRatio     = 0.40
valRatio       = 0.30
testRatio      = 0.30
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
def ResNet50(windowSizeW, windowSizeH, numClass):
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
    outputs = Dense(numClass, activation='softmax', name='fc{}'.format(numClass))(X)
    
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

# Build model
windowSizeW = feature.shape[1]
windowSizeH = feature.shape[2]
print("Dimension of image: " + str(windowSizeW) + "x" + str(windowSizeH))
model = ResNet50(windowSizeW, windowSizeH, numClass)
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
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
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
print('Resnet-50 Classification Report')        
print(tabulate(table, headers, tablefmt="presto"))
print('\n')

with open(fileNameTable, 'w') as f:
    f.write('Classification Resnet-50 Report\n')
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