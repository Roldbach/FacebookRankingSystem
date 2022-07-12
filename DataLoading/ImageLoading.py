import pandas as pd

from Configuration import imageLoadingPath

def loadImageLoading(imageLoadingPath:str=imageLoadingPath) -> pd.DataFrame:
    '''
        Return a data frame which contains:
        1. image ID
        2. label
        3. the file path to load the pixel array

    Argument:
        imageLoadingPath: str, file path of the image loading information
    
    Return:
        imageLoading: pd.DataFrame, the image loading information
    '''
    return pd.read_csv(imageLoadingPath)

