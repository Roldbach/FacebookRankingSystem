import pandas as pd

from Configuration import textDatasetSetting, TFIDFVectorizerSetting
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def splitTrainTest(dataFrame:pd.DataFrame,
                   featureColumn:list[str]=textDatasetSetting["featureColumn"],
                   targetColumn:str=textDatasetSetting["targetColumn"],
                   testSize:float=textDatasetSetting["testSize"]) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    '''
        Split the given dataFrame to the training and test sets separately
    
    Argument:
        dataFrame: pd.DataFrame, the original data frame after data cleaning
        featureColumn: list[str], contains the name of columns that are chosen as features
        targetColumn: str, specifies the column that contains the label
        testSize: float, the proportion of test set within the whole dataset

    Return:
        trainData: pd.DataFrame, the data frame that contains training data
        testData: pd.DataFrame, the data frame that contains test data
        trainLabel: pd.Series, the data series that could be used as training label
        testLabel: pd.Series, the data series that could be used as test label
    '''
    dataset=dataFrame.loc[:, featureColumn]
    label=dataFrame[targetColumn]
    
    trainData, testData, trainLabel, testLabel=train_test_split(dataset, label, test_size=testSize)
    return trainData, testData, trainLabel, testLabel

def transformData(trainData:pd.DataFrame, testData:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
        Transform each column data using TFIDF vectorizer and 
    concatenate all of them at the end

    Argument:
        trainData: pd.DataFrame, the data frame that contains training data
        testData: pd.DataFrame, the data frame that contains test data

    Return:
        train: pd.DataFrame, the transformed training data which could be used for training
        test: pd.DataFrame, the transformed testing data which could be used for testing
    '''
    train=[]
    test=[]
    column=trainData.columns

    for name in column:
        trainColumnDataFrame, testColumnDataFrame=transformColumn(trainData[name], testData[name])
        train.append(trainColumnDataFrame)
        test.append(testColumnDataFrame)

    return pd.concat(train, axis=1), pd.concat(test, axis=1)

def transformColumn(trainColumn:pd.Series, testColumn:pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
        Transform the text data to feature importance using TFIDF
    vectorizer so it could be used for model training

    Argument:
        trainColumn: pd.Series, the original train feature data in a column
        testColumn: pd.Series, the original test feature data in a column
    
    Return:
        trainColumnDataFrame: pd.DataFrame, the data frame containing all filtered features and 
                              their corresponding weight for the training data
        testColumnDataFrame: pd.DataFrame, the data frame containing all filtered features and
                             their corresponding weight for the test data
    '''
    vectorizer=TfidfVectorizer(stop_words=TFIDFVectorizerSetting["stop_words"],
                               max_features=TFIDFVectorizerSetting["max_features"],
                               min_df=TFIDFVectorizerSetting["min_df"],
                               max_df=TFIDFVectorizerSetting["max_df"]).fit(trainColumn)

    trainColumnDataFrame=pd.DataFrame(vectorizer.transform(trainColumn).todense(), columns=vectorizer.get_feature_names_out())
    testColumnDataFrame=pd.DataFrame(vectorizer.transform(testColumn).todense(), columns=vectorizer.get_feature_names_out())

    return trainColumnDataFrame, testColumnDataFrame
