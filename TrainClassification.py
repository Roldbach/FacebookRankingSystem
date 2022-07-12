import numpy as np

from DataLoading.ImageLoading import loadImageLoading
from DataProcessing.ImageProcessing import loadFlatDataset, splitTrainTest
from sklearn.svm import SVC
from sklearn.metrics import classification_report

def loadData() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
        Load the image data for model training

        This function contains the following steps:
        1. Split the image paths into the training part and test part
        2. Load all image arrays within each sub dataset
        3. Flatten each image and stack them together
    
    Return:
        train: np.ndarray, array that contains all training paths
        test: np.ndarray, array that that contains all test paths
        trainLabel: np.ndarray, array that could be used as training label
        testLabel: np.ndarray, array that could be used as test label
    '''
    dataFrame=loadImageLoading()
    trainPath, testPath, trainLabel, testLabel=splitTrainTest(dataFrame)
    return loadFlatDataset(trainPath), loadFlatDataset(testPath), trainLabel, testLabel

if __name__=="__main__":
    train, test, trainLabel, testLabel=loadData()
    model=SVC(gamma=0.001)

    model.fit(train, trainLabel)
    prediction=model.predict(test)

    classification_report(testLabel, prediction)
