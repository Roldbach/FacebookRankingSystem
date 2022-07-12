import pandas as pd

from Configuration import cleanProductName, cleanProductPath, lineTerminator

def loadProduct(cleanProductName:str=cleanProductName, cleanProductPath:str=cleanProductPath, lineTerminator:str=lineTerminator) -> pd.DataFrame:
    '''
        Load and return the processed product information csv file as a data frame
    
    Argument:
        cleanProductName: string, the name of the processed product information
        cleanProductPath: string, file path of processed product information
        lineterminator: string, the terminator used to represent the termination
                        in the csv file

    Return: 
        result: pd.DataFrame, the processed product information
    '''
    return pd.read_csv(cleanProductPath+"/"+cleanProductName+".csv", lineterminator=lineTerminator)