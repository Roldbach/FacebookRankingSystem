import numpy as np
import pandas as pd

from Configuration import imageDatasetSetting
from sklearn.model_selection import train_test_split

def splitTrainTest(dataFrame:pd.DataFrame,
                   pathColumn:str=imageDatasetSetting["pathColumn"],
                   targetColumn:str=imageDatasetSetting["targetColumn"],
                   testSize:float=imageDatasetSetting["testSize"]) -> tuple[pd.Series, pd.Series, np.ndarray, np.ndarray]:
    '''
        Split the given dataFrame to the training and test paths separately,
    which could be used for loading the training and test dataset
    
    Argument:
        dataFrame: pd.DataFrame, the original data frame after data cleaning
        pathColumn: str, specifies the column that contains file paths storing pixel arrays
        targetColumn: str, specifies the column that contains labels
        testSize: float, the proportion of test set within the whole dataset

    Return:
        trainPath: pd.Series, the data series that contains all training paths
        testPath: pd.Series, the data series that contains all test paths
        trainLabel: np.ndarray, array that could be used as training label
        testLabel: np.ndarray, array that could be used as test label
    '''
    dataset=dataFrame[pathColumn]
    label=dataFrame[targetColumn]
    
    trainPath, testPath, trainLabel, testLabel=train_test_split(dataset, label, test_size=testSize)
    return trainPath, testPath, trainLabel.to_numpy(dtype="uint8"), testLabel.to_numpy(dtype="uint8")

def loadFlatDataset(datasetPath:pd.Series) -> np.ndarray:
    '''
        Load all images within the dataset and stack them
    together after flatten
    
    Argument:
        datasetPath: pd.Series, the data series that all paths within a dataset 
    
    Return:
        result: np.ndarray, the stacked image dataset
    '''
    result=[loadFlatImage(path) for path in datasetPath]
    return np.stack(result).astype("float32")

def loadFlatImage(imagePath:str) -> np.ndarray:
    '''
        Load and flatten the pixel array from the given path,
    which could be used for training classification model

    Argument:
        imagePath: str, the file path the stored pixel array
    
    Return:
        result: np.ndarray, the flattened pixel array
    '''
    result=np.load(imagePath)
    return np.ravel(result)

