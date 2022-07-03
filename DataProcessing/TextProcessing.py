import numpy as np
import pandas as pd
import scipy.sparse.csr as csr

from Configuration import countVectorizerSetting, datasetSetting
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split

def splitTrainTest(dataFrame:pd.DataFrame,
                   featureColumn:list[str]=datasetSetting["featureColumn"],
                   targetColumn:str=datasetSetting["targetColumn"],
                   testSize:float=datasetSetting["testSize"]) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    '''
        Split the given dataFrame to the training and test sets separately
    
    Argument:
        dataFrame: pd.DataFrame, the original data frame after data cleaning
        featureColumn: list, contains the name of columns that are chosen as features
        targetColumn: list, specifies the column that contains the label
        testSize: float, the proportion of test set within the whole dataset

    Return:
        train: pd.DataFrame, the data frame that contains training data
        test: pd.DataFrame, the data frame that contains test data
        trainLabel: pd.Series, the data series that could be used as training label
        testLabel: pd.Series, the data series that could be used as test label
    '''
    dataset=dataFrame.loc[:, featureColumn]
    label=dataFrame[targetColumn]
    
    train, test, trainLabel, testLabel=train_test_split(dataset, label, test_size=testSize)
    return train, test, trainLabel, testLabel

def transformColumn(trainColumn:pd.Series, testColumn:pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
        Transform the text data to feature importance using TFIDF
    transformer so it could be used for model training

        The weight could be generated using the following steps:
        1. Construct a CountVectorizer and fit it using the column data
        2. Transform the column data to a sparse representation
        3. Transform the sparse matrix to weight data frame in descending order

    Argument:
        column: pd.Series, the original text data as a column
    
    Return:
        result: pd.DataFrame, the data frame containing all filtered features and 
                              their corresponding weight
    '''
    trainMatrix, testMatrix=CountVectorizeColumn(trainColumn, testColumn)
    trainDataFrame, testDataFrame=TFIDFTransformMatrix(trainMatrix, testMatrix)

    print(trainDataFrame.columns)
    print(testDataFrame.shape)

def CountVectorizeColumn(trainColumn:pd.Series, testColumn:pd.Series) -> tuple[csr.csr_matrix, csr.csr_matrix]:
    '''
        Use CountVectorizer to fit+transform the train column
    and transform the test column so they could both be converted
    to their corresponding sparse representation

    Argument:
        trainColumn: pd.Series, the original train feature data in a column
        testColumn: pd.Series, the original test feature data in a column
    
    Return:
        trainMatrix: scipy.sparse.csr.csr_matrix, the sparse matrix of the train column
        testMatrix: scipy.sparse.csr.csr_matrix, the sparse matrix of the test column
    '''
    vectorizer=CountVectorizer(stop_words=countVectorizerSetting["stop_words"],
                            max_features=countVectorizerSetting["max_features"],
                            min_df=countVectorizerSetting["min_df"],
                            max_df=countVectorizerSetting["max_df"])

    trainMatrix=vectorizer.fit_transform(trainColumn)
    testMatrix=vectorizer.transform(testColumn)

    return trainMatrix, testMatrix

def TFIDFTransformMatrix(trainMatrix:csr.csr_matrix, testMatrix:csr.csr_matrix) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
        Use TFIDF Transformer to fit+transform the train matrix
    and transform the test matrix so they could both be converted
    to data frames containing all features and their corresponding weights

    Argument:
        trainMatrix: scipy.sparse.csr.csr_matrix, the sparse matrix of the train column
        testMatrix: scipy.sparse.csr.csr_matrix, the sparse matrix of the test column

    Return:
        trainDataFrame: pd.DataFrame, the data frame containing all features and 
                                       their corresponding weights in the training set
        testDataFrame: pd.DataFrame, the data frame containing all features and their
                                     corresponding weights in the test set
    '''
    transformer=TfidfTransformer()
    
    trainWeight=transformer.fit_transform(trainMatrix)
    trainWeight=np.asarray(trainWeight.mean(axis=0)).ravel().tolist()
    trainDataFrame=pd.DataFrame({"Word":transformer.get_feature_names_out(), "Weight":trainWeight})

    testWeight=transformer.fit_transform(testMatrix)
    testWeight=np.asarray(testWeight.mean(axis=0)).ravel().tolist()
    testDataFrame=pd.DataFrame({"Word":transformer.get_feature_names_out(), "Weight":testWeight})

    trainWeight=transformer.fit_transform(trainMatrix)
    trainDataFrame=pd.DataFrame(trainWeight.todense(), columns=transformer.get_feature_names_out())

    return trainDataFrame, testDataFrame

