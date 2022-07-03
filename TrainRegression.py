import pandas as pd

from DataCleaning.CleanTabularData import cleanProduct
from DataProcessing.TextProcessing import splitTrainTest, transformData
from sklearn.linear_model import LinearRegression

def loadProduct() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    '''
        Load the product data for model training

        This function contains the following steps:
        1. Clean the original product data
        2. Split the data into the training part and test part
        2. Transform each column in each part to its corresonding data frame
        3. Concatenate all data frames to form the final dataset
    
    Return:
        train: pd.DataFrame, the transformed training data which could be used for training
        test: pd.DataFrame, the transformed testing data which could be used for testing
        trainLabel: pd.Series, the data series that could be used as training label
        testLabel: pd.Series, the data series that could be used as test label
    '''
    data=cleanProduct()
    trainData, testData, trainLabel, testLabel=splitTrainTest(data)
    train, test=transformData(trainData, testData)

    return train, test, trainLabel, testLabel

if __name__=="__main__":
    train, test, trainLabel, testLabel=loadProduct()

    model=LinearRegression().fit(train, trainLabel)
    score=model.score(test, testLabel)

    print("The R2 score of the linear regression model is: ", score)

