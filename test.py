import os

import numpy as np
import pandas as pd
from DataCleaning.CleanImageData import cleanImage

from Configuration import *
from DataCleaning.CleanTabularData import cleanProduct
from DataLoading.TextLoading import loadProduct
from DataProcessing.TextProcessing import splitTrainTest, transformColumn, transformData

from sklearn.feature_extraction.text import TfidfVectorizer

