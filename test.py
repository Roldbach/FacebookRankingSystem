import pandas as pd

from DataCleaning.CleanTabularData import cleanProduct
from DataProcessing.TextProcessing import splitTrainTest, transformColumn, transformData

from sklearn.feature_extraction.text import TfidfVectorizer

test=cleanProduct()
trainData, testData, trainLabel, testLabel=splitTrainTest(test)
train, test=transformData(trainData, testData)

print(train.columns)
