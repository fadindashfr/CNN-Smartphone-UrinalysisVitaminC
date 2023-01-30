# Read Data
import numpy as np
import h5py
import itertools as it

def loadData(hdf5File):
    hdf5_data = h5py.File(hdf5File)
    print(hdf5_data.keys())

    feature = hdf5_data.get('feature')
    feature = np.array(feature)

    feature = np.transpose(feature, axes=[3, 2, 1, 0])

    target = hdf5_data.get('target')
    target = np.array(target)
    target = np.transpose(target, axes=[1, 0])

    return feature, target


from sklearn.model_selection import train_test_split

def splitTrainValTest(X, Y, trainRatio=0.7, valRatio=0.15, testRatio=0.15, randomState=345):
    # train is now 75% of the entire data set
    # the _junk suffix means that we drop that variable completely
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1 - trainRatio, random_state=randomState)

    # test is now 10% of the initial data set
    # validation is now 15% of the initial data set
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=testRatio/(testRatio + valRatio), random_state=randomState) 

    return x_train, x_val, x_test, y_train, y_val, y_test