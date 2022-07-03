import pandas as pd

from DataCleaning.CleanTabularData import cleanProduct
from DataProcessing.TextProcessing import splitTrainTest, transformColumn

from sklearn.feature_extraction.text import TfidfVectorizer

test=cleanProduct()
train, test, trainLabel, testLabel=splitTrainTest(test)


